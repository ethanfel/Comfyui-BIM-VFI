import os
import glob
import logging
import shutil
import subprocess
import tempfile
import torch
import folder_paths
from comfy.utils import ProgressBar

from .inference import BiMVFIModel, EMAVFIModel, SGMVFIModel, GIMMVFIModel, FlashVSRModel
from .bim_vfi_arch import clear_backwarp_cache
from .ema_vfi_arch import clear_warp_cache as clear_ema_warp_cache
from .sgm_vfi_arch import clear_warp_cache as clear_sgm_warp_cache
from .gimm_vfi_arch import clear_gimm_caches

logger = logging.getLogger("Tween")

# Google Drive file ID for the pretrained BIM-VFI model
GDRIVE_FILE_ID = "18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC"
MODEL_FILENAME = "bim_vfi.pth"

# Google Drive folder ID for EMA-VFI pretrained models
EMA_GDRIVE_FOLDER_ID = "16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o"
EMA_DEFAULT_MODEL = "ours_t.pkl"

# Register model folders with ComfyUI
MODEL_DIR = os.path.join(folder_paths.models_dir, "bim-vfi")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

EMA_MODEL_DIR = os.path.join(folder_paths.models_dir, "ema-vfi")
if not os.path.exists(EMA_MODEL_DIR):
    os.makedirs(EMA_MODEL_DIR, exist_ok=True)

# Google Drive folder ID for SGM-VFI pretrained models
SGM_GDRIVE_FOLDER_ID = "1S5O6W0a7XQDHgBtP9HnmoxYEzWBIzSJq"
SGM_DEFAULT_MODEL = "ours-1-2-points.pkl"

SGM_MODEL_DIR = os.path.join(folder_paths.models_dir, "sgm-vfi")
if not os.path.exists(SGM_MODEL_DIR):
    os.makedirs(SGM_MODEL_DIR, exist_ok=True)

# GIMM-VFI
GIMM_HF_REPO = "Kijai/GIMM-VFI_safetensors"
GIMM_AVAILABLE_MODELS = [
    "gimmvfi_r_arb_lpips_fp32.safetensors",
    "gimmvfi_f_arb_lpips_fp32.safetensors",
]

GIMM_MODEL_DIR = os.path.join(folder_paths.models_dir, "gimm-vfi")
if not os.path.exists(GIMM_MODEL_DIR):
    os.makedirs(GIMM_MODEL_DIR, exist_ok=True)


def get_available_models():
    """List available checkpoint files in the bim-vfi model directory."""
    models = []
    if os.path.isdir(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith((".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(MODEL_FILENAME)  # Will trigger auto-download
    return sorted(models)


def download_model_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the BIM-VFI model. "
            "Install it with: pip install gdown"
        )
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading BIM-VFI model to {dest_path}...")
    gdown.download(url, dest_path, quiet=False)
    if not os.path.exists(dest_path):
        raise RuntimeError(f"Failed to download model to {dest_path}")
    logger.info("Download complete.")


class LoadBIMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_models(), {
                    "default": MODEL_FILENAME,
                    "tooltip": "Checkpoint file from models/bim-vfi/. Auto-downloads on first use if missing.",
                }),
                "auto_pyr_level": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically select pyramid level based on input resolution: <540p=3, 540p=5, 1080p=6, 4K=7. Disable to use manual pyr_level.",
                }),
                "pyr_level": ("INT", {
                    "default": 3, "min": 3, "max": 7, "step": 1,
                    "tooltip": "Manual pyramid levels for coarse-to-fine processing. Only used when auto_pyr_level is disabled. More levels = captures larger motion but slower.",
                }),
            }
        }

    RETURN_TYPES = ("BIM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/BIM-VFI"

    def load_model(self, model_path, auto_pyr_level, pyr_level):
        full_path = os.path.join(MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_model_from_gdrive(GDRIVE_FILE_ID, full_path)

        wrapper = BiMVFIModel(
            checkpoint_path=full_path,
            pyr_level=pyr_level,
            auto_pyr_level=auto_pyr_level,
            device="cpu",
        )

        mode = "auto" if auto_pyr_level else f"manual ({pyr_level})"
        logger.info(f"BIM-VFI model loaded (pyr_level={mode})")
        return (wrapper,)


class BIMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("BIM_VFI_MODEL", {
                    "tooltip": "BIM-VFI model from the Load BIM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower. Ignored when all_on_gpu is enabled.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses ~200MB VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full. Recommended: 1 for 8GB, 2-4 for 24GB, 4-16 for 48GB+.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead. Result is identical to processing all at once.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/BIM-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames.

        Args:
            frames: [N, C, H, W] tensor on storage_device
            step_ref: list with single int, mutable counter for progress bar
        Returns:
            Interpolated frames as [M, C, H, W] tensor on storage_device
        """
        for pass_idx in range(num_passes):
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                if not keep_device:
                    model.to(device)

                mids = model.interpolate_batch(frames0, frames1, time_step=0.5)
                mids = mids.to(storage_device)

                if not keep_device:
                    model.to("cpu")

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if not all_on_gpu and pairs_since_clear >= clear_cache_after_n_frames and torch.cuda.is_available():
                    clear_backwarp_cache()
                    torch.cuda.empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            if not all_on_gpu and torch.cuda.is_available():
                clear_backwarp_cache()
                torch.cuda.empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size):
        if images.shape[0] < 2:
            return (images,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_passes = {2: 1, 4: 2, 8: 3}[multiplier]

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)
        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        return (result,)


class BIMVFISegmentInterpolate(BIMVFIInterpolate):
    """Process a numbered segment of the input batch.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = BIMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "BIM_VFI_MODEL")
    RETURN_NAMES = ("images", "model")
    FUNCTION = "interpolate"
    CATEGORY = "video/BIM-VFI"

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size):
        total_input = images.shape[0]

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            # Past the end — return empty single frame + model
            return (images[:1], model)

        segment_images = images[start:end]
        is_continuation = segment_index > 0

        # Delegate to the parent interpolation logic
        (result,) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]  # skip duplicate boundary frame

        return (result, model)


class TweenConcatVideos:
    """Concatenate segment video files into a single video using ffmpeg.

    Connect the model output from the last Segment Interpolate node to ensure
    this runs only after all segments have been saved to disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*", {
                    "tooltip": "Connect from the last Segment Interpolate's model output (any model type). "
                               "This ensures concatenation runs only after all segments are saved.",
                }),
                "output_directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing the segment video files. "
                               "Leave empty to use ComfyUI's default output directory. "
                               "Relative paths are resolved from the output directory.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "segment",
                    "tooltip": "Filename prefix used when saving segments with VHS Video Combine. "
                               "Matches files like segment_00001.mp4, segment_00002.mp4, etc.",
                }),
                "output_filename": ("STRING", {
                    "default": "final_video.mp4",
                    "tooltip": "Name of the concatenated output file. Saved in the same directory.",
                }),
                "delete_segments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Delete the individual segment files after successful concatenation. "
                               "Useful to avoid leftover files that would pollute the next run.",
                }),
                "preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show the concatenated video as a preview on the node. "
                               "Disable to skip the preview widget.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "concat"
    CATEGORY = "video/Tween"

    @staticmethod
    def _find_ffmpeg():
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except ImportError:
                pass
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg or pip install imageio-ffmpeg."
            )
        return ffmpeg_path

    def concat(self, model, output_directory, filename_prefix, output_filename, delete_segments, preview):
        # Resolve output directory — empty or relative paths are relative to ComfyUI output
        comfy_output = folder_paths.get_output_directory()
        out_dir = output_directory.strip()
        if not out_dir:
            out_dir = comfy_output
        elif not os.path.isabs(out_dir):
            out_dir = os.path.join(comfy_output, out_dir)

        if not os.path.isdir(out_dir):
            raise ValueError(f"Output directory does not exist: {out_dir}")

        # Find segment files matching the prefix
        safe_prefix = glob.escape(filename_prefix)
        segments = []
        for ext in ("mp4", "webm", "mkv"):
            segments.extend(
                glob.glob(os.path.join(out_dir, f"{safe_prefix}_*.{ext}"))
            )
        segments.sort()

        if not segments:
            raise FileNotFoundError(
                f"No segment files found matching '{filename_prefix}_*' "
                f"in {out_dir}"
            )

        logger.info(f"Found {len(segments)} segment(s) to concatenate")

        # Write ffmpeg concat list to a temp file
        fd, concat_list_path = tempfile.mkstemp(suffix=".txt", prefix="bimvfi_concat_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("ffconcat version 1.0\n")
                for seg in segments:
                    # ffconcat escaping: \ -> \\, ' -> \'
                    escaped = os.path.abspath(seg).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped}'\n")

            output_path = os.path.join(out_dir, output_filename)
            ffmpeg = self._find_ffmpeg()

            cmd = [
                ffmpeg,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                output_path,
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            logger.info(f"Concatenated video saved to {output_path}")

            if delete_segments:
                for seg in segments:
                    try:
                        os.remove(seg)
                    except OSError as e:
                        logger.warning(f"Failed to delete segment {seg}: {e}")
                logger.info(f"Deleted {len(segments)} segment file(s)")
        finally:
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)

        result = {"result": (output_path,)}

        if preview:
            # Preview only works when the file is inside ComfyUI's output tree
            abs_out = os.path.abspath(out_dir)
            abs_comfy = os.path.abspath(comfy_output)
            if abs_out.startswith(abs_comfy + os.sep) or abs_out == abs_comfy:
                subfolder = os.path.relpath(abs_out, abs_comfy) if abs_out != abs_comfy else ""
                result["ui"] = {
                    "gifs": [{
                        "filename": os.path.basename(output_path),
                        "subfolder": subfolder,
                        "type": "output",
                        "format": "video/mp4",
                    }]
                }
            else:
                logger.warning(
                    f"Video preview skipped: {out_dir} is outside ComfyUI output directory"
                )

        return result


# ---------------------------------------------------------------------------
# EMA-VFI nodes
# ---------------------------------------------------------------------------

def get_available_ema_models():
    """List available checkpoint files in the ema-vfi model directory."""
    models = []
    if os.path.isdir(EMA_MODEL_DIR):
        for f in os.listdir(EMA_MODEL_DIR):
            if f.endswith((".pkl", ".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(EMA_DEFAULT_MODEL)  # Will trigger auto-download
    return sorted(models)


def download_ema_model_from_gdrive(folder_id, dest_path):
    """Download EMA-VFI model from Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the EMA-VFI model. "
            "Install it with: pip install gdown"
        )
    filename = os.path.basename(dest_path)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Downloading {filename} from Google Drive folder to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gdown.download_folder(url, output=os.path.dirname(dest_path), quiet=False, remaining_ok=True)
    if not os.path.exists(dest_path):
        raise RuntimeError(
            f"Failed to download {filename}. Please download manually from "
            f"https://drive.google.com/drive/folders/{folder_id} "
            f"and place it in {os.path.dirname(dest_path)}"
        )
    logger.info("Download complete.")


class LoadEMAVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_ema_models(), {
                    "default": EMA_DEFAULT_MODEL,
                    "tooltip": "Checkpoint file from models/ema-vfi/. Auto-downloads on first use if missing. "
                               "Variant (large/small) and timestep support are auto-detected from filename.",
                }),
                "tta": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Test-time augmentation: flip input and average with unflipped result. "
                               "~2x slower but slightly better quality. Recommended for large model only.",
                }),
            }
        }

    RETURN_TYPES = ("EMA_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/EMA-VFI"

    def load_model(self, model_path, tta):
        full_path = os.path.join(EMA_MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_ema_model_from_gdrive(EMA_GDRIVE_FOLDER_ID, full_path)

        wrapper = EMAVFIModel(
            checkpoint_path=full_path,
            variant="auto",
            tta=tta,
            device="cpu",
        )

        t_mode = "arbitrary" if wrapper.supports_arbitrary_t else "fixed (0.5)"
        logger.info(f"EMA-VFI model loaded (variant={wrapper.variant_name}, timestep={t_mode}, tta={tta})")
        return (wrapper,)


class EMAVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("EMA_VFI_MODEL", {
                    "tooltip": "EMA-VFI model from the Load EMA-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower. Ignored when all_on_gpu is enabled.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/EMA-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames."""
        for pass_idx in range(num_passes):
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                if not keep_device:
                    model.to(device)

                mids = model.interpolate_batch(frames0, frames1, time_step=0.5)
                mids = mids.to(storage_device)

                if not keep_device:
                    model.to("cpu")

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if not all_on_gpu and pairs_since_clear >= clear_cache_after_n_frames and torch.cuda.is_available():
                    clear_ema_warp_cache()
                    torch.cuda.empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            if not all_on_gpu and torch.cuda.is_available():
                clear_ema_warp_cache()
                torch.cuda.empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size):
        if images.shape[0] < 2:
            return (images,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_passes = {2: 1, 4: 2, 8: 3}[multiplier]

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)
        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        return (result,)


class EMAVFISegmentInterpolate(EMAVFIInterpolate):
    """Process a numbered segment of the input batch for EMA-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = EMAVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "EMA_VFI_MODEL")
    RETURN_NAMES = ("images", "model")
    FUNCTION = "interpolate"
    CATEGORY = "video/EMA-VFI"

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size):
        total_input = images.shape[0]

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            # Past the end — return empty single frame + model
            return (images[:1], model)

        segment_images = images[start:end]
        is_continuation = segment_index > 0

        # Delegate to the parent interpolation logic
        (result,) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]  # skip duplicate boundary frame

        return (result, model)


# ---------------------------------------------------------------------------
# SGM-VFI nodes
# ---------------------------------------------------------------------------

def get_available_sgm_models():
    """List available checkpoint files in the sgm-vfi model directory."""
    models = []
    if os.path.isdir(SGM_MODEL_DIR):
        for f in os.listdir(SGM_MODEL_DIR):
            if f.endswith((".pkl", ".pth", ".pt", ".ckpt", ".safetensors")):
                models.append(f)
    if not models:
        models.append(SGM_DEFAULT_MODEL)  # Will trigger auto-download
    return sorted(models)


def download_sgm_model_from_gdrive(folder_id, dest_path):
    """Download SGM-VFI model from Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to auto-download the SGM-VFI model. "
            "Install it with: pip install gdown"
        )
    filename = os.path.basename(dest_path)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"Downloading {filename} from Google Drive folder to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    gdown.download_folder(url, output=os.path.dirname(dest_path), quiet=False, remaining_ok=True)
    if not os.path.exists(dest_path):
        raise RuntimeError(
            f"Failed to download {filename}. Please download manually from "
            f"https://drive.google.com/drive/folders/{folder_id} "
            f"and place it in {os.path.dirname(dest_path)}"
        )
    logger.info("Download complete.")


class LoadSGMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_sgm_models(), {
                    "default": SGM_DEFAULT_MODEL,
                    "tooltip": "Checkpoint file from models/sgm-vfi/. Auto-downloads on first use if missing. "
                               "Variant (base/small) is auto-detected from filename.",
                }),
                "tta": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Test-time augmentation: flip input and average with unflipped result. "
                               "~2x slower but slightly better quality.",
                }),
                "num_key_points": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Sparsity of global matching. 0.0 = global matching everywhere (slower, better for large motion). "
                               "Higher = sparser keypoints (faster). Default 0.5 is a good balance.",
                }),
            }
        }

    RETURN_TYPES = ("SGM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/SGM-VFI"

    def load_model(self, model_path, tta, num_key_points):
        full_path = os.path.join(SGM_MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_sgm_model_from_gdrive(SGM_GDRIVE_FOLDER_ID, full_path)

        wrapper = SGMVFIModel(
            checkpoint_path=full_path,
            variant="auto",
            num_key_points=num_key_points,
            tta=tta,
            device="cpu",
        )

        logger.info(f"SGM-VFI model loaded (variant={wrapper.variant_name}, num_key_points={num_key_points}, tta={tta})")
        return (wrapper,)


class SGMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("SGM_VFI_MODEL", {
                    "tooltip": "SGM-VFI model from the Load SGM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. 2x=one interpolation pass, 4x=two recursive passes, 8x=three. Higher = more frames but longer processing.",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower. Ignored when all_on_gpu is enabled.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously. Higher = faster but uses more VRAM. Start with 1, increase until VRAM is full.",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/SGM-VFI"

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Run all interpolation passes on a chunk of frames."""
        for pass_idx in range(num_passes):
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                if not keep_device:
                    model.to(device)

                mids = model.interpolate_batch(frames0, frames1, time_step=0.5)
                mids = mids.to(storage_device)

                if not keep_device:
                    model.to("cpu")

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if not all_on_gpu and pairs_since_clear >= clear_cache_after_n_frames and torch.cuda.is_available():
                    clear_sgm_warp_cache()
                    torch.cuda.empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            if not all_on_gpu and torch.cuda.is_available():
                clear_sgm_warp_cache()
                torch.cuda.empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for a given input frame count."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size):
        if images.shape[0] < 2:
            return (images,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_passes = {2: 1, 4: 2, 8: 3}[multiplier]

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        # Calculate total progress steps across all chunks
        total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            chunk_result = self._interpolate_frames(
                chunk_frames, model, num_passes, batch_size,
                device, storage_device, keep_device, all_on_gpu,
                clear_cache_after_n_frames, pbar, step_ref,
            )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)
        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        return (result,)


class SGMVFISegmentInterpolate(SGMVFIInterpolate):
    """Process a numbered segment of the input batch for SGM-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = SGMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "SGM_VFI_MODEL")
    RETURN_NAMES = ("images", "model")
    FUNCTION = "interpolate"
    CATEGORY = "video/SGM-VFI"

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames,
                    keep_device, all_on_gpu, batch_size, chunk_size,
                    segment_index, segment_size):
        total_input = images.shape[0]

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            # Past the end — return empty single frame + model
            return (images[:1], model)

        segment_images = images[start:end]
        is_continuation = segment_index > 0

        # Delegate to the parent interpolation logic
        (result,) = super().interpolate(
            segment_images, model, multiplier, clear_cache_after_n_frames,
            keep_device, all_on_gpu, batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]  # skip duplicate boundary frame

        return (result, model)


# ---------------------------------------------------------------------------
# GIMM-VFI nodes
# ---------------------------------------------------------------------------

def get_available_gimm_models():
    """List available GIMM-VFI checkpoint files in the gimm-vfi model directory."""
    models = []
    if os.path.isdir(GIMM_MODEL_DIR):
        for f in os.listdir(GIMM_MODEL_DIR):
            if f.endswith((".safetensors", ".pth", ".pt", ".ckpt")):
                # Exclude flow estimator checkpoints from the model list
                if f.startswith(("raft-", "flowformer_")):
                    continue
                models.append(f)
    if not models:
        models = list(GIMM_AVAILABLE_MODELS)
    return sorted(models)


def download_gimm_model(filename, dest_dir):
    """Download a GIMM-VFI file from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to auto-download GIMM-VFI models. "
            "Install it with: pip install huggingface_hub"
        )
    logger.info(f"Downloading {filename} from HuggingFace ({GIMM_HF_REPO})...")
    hf_hub_download(
        repo_id=GIMM_HF_REPO,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        raise RuntimeError(f"Failed to download {filename} to {dest_path}")
    logger.info(f"Downloaded {filename}")


class LoadGIMMVFIModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (get_available_gimm_models(), {
                    "default": GIMM_AVAILABLE_MODELS[0],
                    "tooltip": "Checkpoint file from models/gimm-vfi/. Auto-downloads from HuggingFace on first use. "
                               "RAFT variant (~80MB) or FlowFormer variant (~123MB) auto-detected from filename.",
                }),
                "ds_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.125, "max": 1.0, "step": 0.125,
                    "tooltip": "Downscale factor for internal processing. 1.0 = full resolution. "
                               "Lower values reduce VRAM usage and speed up inference at the cost of quality. "
                               "Try 0.5 for 4K inputs.",
                }),
            }
        }

    RETURN_TYPES = ("GIMM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/GIMM-VFI"

    def load_model(self, model_path, ds_factor):
        full_path = os.path.join(GIMM_MODEL_DIR, model_path)

        # Auto-download main model if missing
        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_gimm_model(model_path, GIMM_MODEL_DIR)

        # Detect and download matching flow estimator
        if "gimmvfi_f" in model_path.lower():
            flow_filename = "flowformer_sintel_fp32.safetensors"
        else:
            flow_filename = "raft-things_fp32.safetensors"

        flow_path = os.path.join(GIMM_MODEL_DIR, flow_filename)
        if not os.path.exists(flow_path):
            logger.info(f"Flow estimator not found, downloading {flow_filename}...")
            download_gimm_model(flow_filename, GIMM_MODEL_DIR)

        wrapper = GIMMVFIModel(
            checkpoint_path=full_path,
            flow_checkpoint_path=flow_path,
            variant="auto",
            ds_factor=ds_factor,
            device="cpu",
        )

        logger.info(f"GIMM-VFI model loaded (variant={wrapper.variant_name}, ds_factor={ds_factor})")
        return (wrapper,)


class GIMMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input image batch. Output frame count: 2x=(2N-1), 4x=(4N-3), 8x=(8N-7).",
                }),
                "model": ("GIMM_VFI_MODEL", {
                    "tooltip": "GIMM-VFI model from the Load GIMM-VFI Model node.",
                }),
                "multiplier": ([2, 4, 8], {
                    "default": 2,
                    "tooltip": "Frame rate multiplier. In single-pass mode, all intermediate frames are generated "
                               "in one forward pass per pair. In recursive mode, uses 2x passes like other models.",
                }),
                "single_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GIMM-VFI's single-pass arbitrary-timestep mode. Generates all intermediate frames "
                               "per pair in one forward pass (no recursive 2x passes). Disable to use the standard "
                               "recursive approach (same as BIM/EMA/SGM).",
                }),
                "clear_cache_after_n_frames": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Clear CUDA cache every N frame pairs to prevent VRAM buildup. Lower = less VRAM but slower. Ignored when all_on_gpu is enabled.",
                }),
                "keep_device": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU between frame pairs. Faster but uses more VRAM constantly. Disable to free VRAM between pairs (slower due to CPU-GPU transfers).",
                }),
                "all_on_gpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Store all intermediate frames on GPU instead of CPU. Much faster (no transfers) but requires enough VRAM for all frames. Recommended for 48GB+ cards.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frame pairs to process simultaneously in recursive mode. Ignored in single-pass mode (pairs are processed one at a time since each generates multiple frames).",
                }),
                "chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Bounds VRAM usage during processing but the full output is still assembled in RAM. To bound RAM, use the Segment Interpolate node instead.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/GIMM-VFI"

    def _interpolate_frames_single_pass(self, frames, model, multiplier,
                                        device, storage_device, keep_device, all_on_gpu,
                                        clear_cache_after_n_frames, pbar, step_ref):
        """Single-pass interpolation using GIMM-VFI's arbitrary timestep capability."""
        num_intermediates = multiplier - 1
        new_frames = []
        num_pairs = frames.shape[0] - 1
        pairs_since_clear = 0

        for i in range(num_pairs):
            frame0 = frames[i:i+1]
            frame1 = frames[i+1:i+2]

            if not keep_device:
                model.to(device)

            mids = model.interpolate_multi(frame0, frame1, num_intermediates)
            mids = [m.to(storage_device) for m in mids]

            if not keep_device:
                model.to("cpu")

            new_frames.append(frames[i:i+1])
            for m in mids:
                new_frames.append(m)

            step_ref[0] += 1
            pbar.update_absolute(step_ref[0])

            pairs_since_clear += 1
            if not all_on_gpu and pairs_since_clear >= clear_cache_after_n_frames and torch.cuda.is_available():
                clear_gimm_caches()
                torch.cuda.empty_cache()
                pairs_since_clear = 0

        new_frames.append(frames[-1:])
        result = torch.cat(new_frames, dim=0)

        if not all_on_gpu and torch.cuda.is_available():
            clear_gimm_caches()
            torch.cuda.empty_cache()

        return result

    def _interpolate_frames(self, frames, model, num_passes, batch_size,
                            device, storage_device, keep_device, all_on_gpu,
                            clear_cache_after_n_frames, pbar, step_ref):
        """Recursive 2x interpolation (standard approach, same as other models)."""
        for pass_idx in range(num_passes):
            new_frames = []
            num_pairs = frames.shape[0] - 1
            pairs_since_clear = 0

            for i in range(0, num_pairs, batch_size):
                batch_end = min(i + batch_size, num_pairs)
                actual_batch = batch_end - i

                frames0 = frames[i:batch_end]
                frames1 = frames[i + 1:batch_end + 1]

                if not keep_device:
                    model.to(device)

                mids = model.interpolate_batch(frames0, frames1, time_step=0.5)
                mids = mids.to(storage_device)

                if not keep_device:
                    model.to("cpu")

                for j in range(actual_batch):
                    new_frames.append(frames[i + j:i + j + 1])
                    new_frames.append(mids[j:j+1])

                step_ref[0] += actual_batch
                pbar.update_absolute(step_ref[0])

                pairs_since_clear += actual_batch
                if not all_on_gpu and pairs_since_clear >= clear_cache_after_n_frames and torch.cuda.is_available():
                    clear_gimm_caches()
                    torch.cuda.empty_cache()
                    pairs_since_clear = 0

            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            if not all_on_gpu and torch.cuda.is_available():
                clear_gimm_caches()
                torch.cuda.empty_cache()

        return frames

    @staticmethod
    def _count_steps(num_frames, num_passes):
        """Count total interpolation steps for recursive mode."""
        n = num_frames
        total = 0
        for _ in range(num_passes):
            total += n - 1
            n = 2 * n - 1
        return total

    def interpolate(self, images, model, multiplier, single_pass,
                    clear_cache_after_n_frames, keep_device, all_on_gpu,
                    batch_size, chunk_size):
        if images.shape[0] < 2:
            return (images,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not single_pass:
            num_passes = {2: 1, 4: 2, 8: 3}[multiplier]

        if all_on_gpu:
            keep_device = True

        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        all_frames = images.permute(0, 3, 1, 2).to(storage_device)
        total_input = all_frames.shape[0]

        # Build chunk boundaries (1-frame overlap between consecutive chunks)
        if chunk_size < 2 or chunk_size >= total_input:
            chunks = [(0, total_input)]
        else:
            chunks = []
            start = 0
            while start < total_input - 1:
                end = min(start + chunk_size, total_input)
                chunks.append((start, end))
                start = end - 1  # overlap by 1 frame
                if end == total_input:
                    break

        # Calculate total progress steps across all chunks
        if single_pass:
            total_steps = sum(ce - cs - 1 for cs, ce in chunks)
        else:
            total_steps = sum(self._count_steps(ce - cs, num_passes) for cs, ce in chunks)
        pbar = ProgressBar(total_steps)
        step_ref = [0]

        if keep_device:
            model.to(device)

        result_chunks = []
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
            chunk_frames = all_frames[chunk_start:chunk_end].clone()

            if single_pass:
                chunk_result = self._interpolate_frames_single_pass(
                    chunk_frames, model, multiplier,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )
            else:
                chunk_result = self._interpolate_frames(
                    chunk_frames, model, num_passes, batch_size,
                    device, storage_device, keep_device, all_on_gpu,
                    clear_cache_after_n_frames, pbar, step_ref,
                )

            # Skip first frame of subsequent chunks (duplicate of previous chunk's last frame)
            if chunk_idx > 0:
                chunk_result = chunk_result[1:]

            # Move completed chunk to CPU to bound memory when chunking
            if len(chunks) > 1:
                chunk_result = chunk_result.cpu()

            result_chunks.append(chunk_result)

        result = torch.cat(result_chunks, dim=0)
        # Convert back to ComfyUI [B, H, W, C], on CPU
        result = result.cpu().permute(0, 2, 3, 1)
        return (result,)


class GIMMVFISegmentInterpolate(GIMMVFIInterpolate):
    """Process a numbered segment of the input batch for GIMM-VFI.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through output forces sequential execution so each segment
    saves and frees from RAM before the next starts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = GIMMVFIInterpolate.INPUT_TYPES()
        base["required"]["segment_index"] = ("INT", {
            "default": 0, "min": 0, "max": 10000, "step": 1,
            "tooltip": "Which segment to process (0-based). Bounds RAM by only producing this segment's output frames, "
                       "unlike chunk_size which bounds VRAM but still assembles the full output in RAM. "
                       "Chain the model output to the next Segment Interpolate to force sequential execution.",
        })
        base["required"]["segment_size"] = ("INT", {
            "default": 500, "min": 2, "max": 10000, "step": 1,
            "tooltip": "Number of input frames per segment. Adjacent segments overlap by 1 frame for seamless stitching. "
                       "Smaller = less peak RAM per segment. Save each segment's output to disk before the next runs.",
        })
        return base

    RETURN_TYPES = ("IMAGE", "GIMM_VFI_MODEL")
    RETURN_NAMES = ("images", "model")
    FUNCTION = "interpolate"
    CATEGORY = "video/GIMM-VFI"

    def interpolate(self, images, model, multiplier, single_pass,
                    clear_cache_after_n_frames, keep_device, all_on_gpu,
                    batch_size, chunk_size, segment_index, segment_size):
        total_input = images.shape[0]

        # Compute segment boundaries (1-frame overlap)
        start = segment_index * (segment_size - 1)
        end = min(start + segment_size, total_input)

        if start >= total_input - 1:
            # Past the end — return empty single frame + model
            return (images[:1], model)

        segment_images = images[start:end]
        is_continuation = segment_index > 0

        # Delegate to the parent interpolation logic
        (result,) = super().interpolate(
            segment_images, model, multiplier, single_pass,
            clear_cache_after_n_frames, keep_device, all_on_gpu,
            batch_size, chunk_size,
        )

        if is_continuation:
            result = result[1:]  # skip duplicate boundary frame

        return (result, model)


# ---------------------------------------------------------------------------
# FlashVSR nodes (4x video super-resolution)
# ---------------------------------------------------------------------------

FLASHVSR_HF_REPO = "1038lab/FlashVSR"
FLASHVSR_REQUIRED_FILES = [
    "FlashVSR1_1.safetensors",
    "Wan2.1_VAE.safetensors",
    "LQ_proj_in.safetensors",
    "TCDecoder.safetensors",
    "Prompt.safetensors",
]

# Check common locations so we reuse models from 1038lab/ComfyUI-FlashVSR
FLASHVSR_MODEL_DIR = None
for _dirname in ("FlashVSR", "flashvsr"):
    _candidate = os.path.join(folder_paths.models_dir, _dirname)
    if os.path.isdir(_candidate) and all(
        os.path.exists(os.path.join(_candidate, f)) for f in FLASHVSR_REQUIRED_FILES
    ):
        FLASHVSR_MODEL_DIR = _candidate
        break
if FLASHVSR_MODEL_DIR is None:
    # Default to "FlashVSR" (matches 1038lab convention)
    FLASHVSR_MODEL_DIR = os.path.join(folder_paths.models_dir, "FlashVSR")


def download_flashvsr_models(model_dir):
    """Download FlashVSR checkpoints from HuggingFace if missing."""
    from huggingface_hub import snapshot_download

    missing = [f for f in FLASHVSR_REQUIRED_FILES
               if not os.path.exists(os.path.join(model_dir, f))]
    if not missing:
        return

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"[FlashVSR] Missing files: {', '.join(missing)}. Downloading from HuggingFace...")
    snapshot_download(
        repo_id=FLASHVSR_HF_REPO,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    still_missing = [f for f in FLASHVSR_REQUIRED_FILES
                     if not os.path.exists(os.path.join(model_dir, f))]
    if still_missing:
        raise FileNotFoundError(
            f"[FlashVSR] Failed to download: {', '.join(still_missing)}. "
            f"Please download manually from https://huggingface.co/{FLASHVSR_HF_REPO}"
        )
    logger.info("[FlashVSR] All checkpoints downloaded successfully.")


class _FlashVSRProgressBar:
    """Wrap an iterable with a ComfyUI ProgressBar."""

    def __init__(self, total, pbar, step_ref):
        self.total = total
        self.pbar = pbar
        self.step_ref = step_ref

    def __call__(self, iterable):
        return self._Wrapper(iterable, self.pbar, self.step_ref)

    class _Wrapper:
        def __init__(self, iterable, pbar, step_ref):
            self.iterable = iterable
            self.pbar = pbar
            self.step_ref = step_ref
            self._iter = iter(iterable)

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self._iter)
            self.step_ref[0] += 1
            self.pbar.update_absolute(self.step_ref[0])
            return val

        def __len__(self):
            return len(self.iterable)


class LoadFlashVSRModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": "Pipeline mode. Tiny: fast TCDecoder decode. "
                               "Tiny-long: streaming TCDecoder, lowest VRAM for long videos. "
                               "Full: standard VAE decode, highest quality but more VRAM.",
                }),
                "precision": (["bf16", "fp16"], {
                    "default": "bf16",
                    "tooltip": "Model precision. BF16 is faster on modern GPUs. FP16 for older GPUs.",
                }),
            }
        }

    RETURN_TYPES = ("FLASHVSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/FlashVSR"

    def load_model(self, mode, precision):
        download_flashvsr_models(FLASHVSR_MODEL_DIR)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        wrapper = FlashVSRModel(
            model_dir=FLASHVSR_MODEL_DIR,
            mode=mode,
            device=device,
            dtype=dtype,
        )

        logger.info(f"[FlashVSR] Model loaded (mode={mode}, precision={precision})")
        return (wrapper,)


class FlashVSRUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input video frames. Minimum 21 frames required.",
                }),
                "model": ("FLASHVSR_MODEL", {
                    "tooltip": "FlashVSR model from the Load FlashVSR Model node.",
                }),
                "scale": ("INT", {
                    "default": 4, "min": 2, "max": 4, "step": 2,
                    "tooltip": "Upscaling factor. 4x is the native resolution; 2x is supported but less optimized.",
                }),
                "frame_chunk_size": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Process frames in chunks of this size to bound VRAM (0=all at once). "
                               "Each chunk must be >= 21 frames. Recommended: 33 (4x8+1) or 65 (8x8+1).",
                }),
                "tiled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE tiled decode. Reduces VRAM usage significantly.",
                }),
                "tile_size_h": ("INT", {
                    "default": 60, "min": 16, "max": 256, "step": 4,
                    "tooltip": "VAE tile height (in latent space). Larger = faster but more VRAM.",
                }),
                "tile_size_w": ("INT", {
                    "default": 104, "min": 16, "max": 256, "step": 4,
                    "tooltip": "VAE tile width (in latent space). Larger = faster but more VRAM.",
                }),
                "topk_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Sparse attention ratio. Higher = faster but may lose fine detail.",
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "KV cache ratio. Higher = better quality, more VRAM.",
                }),
                "local_range": ([9, 11], {
                    "default": 9,
                    "tooltip": "Local attention window. 9=sharper details, 11=more temporal stability.",
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply color correction to prevent color shifts from the diffusion process.",
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload DiT to CPU before VAE decode. Saves VRAM but slower.",
                }),
                "seed": ("INT", {
                    "default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for the diffusion process.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = "video/FlashVSR"

    def upscale(self, images, model, scale, frame_chunk_size,
                tiled, tile_size_h, tile_size_w,
                topk_ratio, kv_ratio, local_range,
                color_fix, unload_dit, seed):
        num_frames = images.shape[0]
        if num_frames < FlashVSRModel.MIN_FRAMES:
            raise ValueError(
                f"FlashVSR requires at least {FlashVSRModel.MIN_FRAMES} frames, got {num_frames}"
            )

        tile_size = (tile_size_h, tile_size_w)

        # Build frame chunks
        if frame_chunk_size < FlashVSRModel.MIN_FRAMES or frame_chunk_size >= num_frames:
            chunks = [(0, num_frames)]
        else:
            chunks = []
            start = 0
            while start < num_frames:
                end = min(start + frame_chunk_size, num_frames)
                chunks.append((start, end))
                if end == num_frames:
                    break
                start = end
            # If the last chunk is too small, merge it into the previous one
            if len(chunks) > 1 and (chunks[-1][1] - chunks[-1][0]) < FlashVSRModel.MIN_FRAMES:
                prev_start = chunks[-2][0]
                last_end = chunks[-1][1]
                chunks = chunks[:-2]
                chunks.append((prev_start, last_end))

        # Estimate total pipeline steps for progress bar
        # Mirrors _pad_video_5d: pad to num_frames % 8 == 1, min 25
        total_steps = 0
        for cs, ce in chunks:
            nf = max(ce - cs, 25)
            remainder = (nf - 1) % 8
            if remainder != 0:
                nf += 8 - remainder
            total_steps += max(1, (nf - 1) // 8 - 2)

        pbar = ProgressBar(total_steps)
        step_ref = [0]
        progress = _FlashVSRProgressBar(total_steps, pbar, step_ref)

        model.load_to_device()

        result_chunks = []
        for chunk_start, chunk_end in chunks:
            chunk_frames = images[chunk_start:chunk_end]

            chunk_result = model.upscale(
                chunk_frames,
                scale=scale, tiled=tiled, tile_size=tile_size,
                topk_ratio=topk_ratio, kv_ratio=kv_ratio,
                local_range=local_range, color_fix=color_fix,
                unload_dit=unload_dit, seed=seed,
                progress_bar_cmd=progress,
            )
            result_chunks.append(chunk_result)
            model.clear_caches()

        model.offload()
        from .flashvsr_arch.models.utils import clean_vram
        clean_vram()

        return (torch.cat(result_chunks, dim=0),)


class FlashVSRSegmentUpscale:
    """Process a numbered segment with temporal overlap and crossfade blending.

    Chain multiple instances with Save nodes between them to bound peak RAM.
    The model pass-through forces sequential execution so each segment
    saves and frees RAM before the next starts.

    Crossfade blending within the overlap region:
    - First (overlap - blend) frames: warmup only, discarded from output
    - Last blend frames: linear alpha crossfade with previous segment's tail
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Full input video frames. Minimum 21 frames required.",
                }),
                "model": ("FLASHVSR_MODEL", {
                    "tooltip": "FlashVSR model from Load FlashVSR Model. "
                               "Chain the model output to the next segment node for sequential execution.",
                }),
                "segment_index": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Which segment to process (0-based).",
                }),
                "segment_size": ("INT", {
                    "default": 100, "min": 21, "max": 10000, "step": 1,
                    "tooltip": "Number of input frames per segment.",
                }),
                "overlap_frames": ("INT", {
                    "default": 8, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Number of overlapping frames between adjacent segments. "
                               "These frames provide temporal context and crossfade blending.",
                }),
                "blend_frames": ("INT", {
                    "default": 4, "min": 0, "max": 50, "step": 1,
                    "tooltip": "Number of frames within the overlap region to crossfade. "
                               "Must be <= overlap_frames. The rest of the overlap is warmup (discarded).",
                }),
                "scale": ("INT", {
                    "default": 4, "min": 2, "max": 4, "step": 2,
                    "tooltip": "Upscaling factor.",
                }),
                "tiled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE tiled decode.",
                }),
                "tile_size_h": ("INT", {
                    "default": 60, "min": 16, "max": 256, "step": 4,
                }),
                "tile_size_w": ("INT", {
                    "default": 104, "min": 16, "max": 256, "step": 4,
                }),
                "topk_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1,
                }),
                "local_range": ([9, 11], {
                    "default": 9,
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                }),
                "seed": ("INT", {
                    "default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLASHVSR_MODEL")
    RETURN_NAMES = ("images", "model")
    FUNCTION = "upscale"
    CATEGORY = "video/FlashVSR"

    def upscale(self, images, model, segment_index, segment_size,
                overlap_frames, blend_frames, scale,
                tiled, tile_size_h, tile_size_w,
                topk_ratio, kv_ratio, local_range,
                color_fix, unload_dit, seed):
        total_input = images.shape[0]
        blend_frames = min(blend_frames, overlap_frames)

        # Clear stale overlap data from previous workflow runs
        if segment_index == 0:
            model._overlap_tail = None

        # Compute segment boundaries
        stride = segment_size - overlap_frames
        start = segment_index * stride
        end = min(start + segment_size, total_input)

        if start >= total_input:
            # Past the end
            return (images[:1], model)

        # Ensure minimum frame count
        actual_size = end - start
        if actual_size < FlashVSRModel.MIN_FRAMES:
            start = max(0, end - FlashVSRModel.MIN_FRAMES)
            actual_size = end - start

        segment_frames = images[start:end]

        tile_size = (tile_size_h, tile_size_w)

        model.load_to_device()

        result = model.upscale(
            segment_frames,
            scale=scale, tiled=tiled, tile_size=tile_size,
            topk_ratio=topk_ratio, kv_ratio=kv_ratio,
            local_range=local_range, color_fix=color_fix,
            unload_dit=unload_dit, seed=seed,
        )

        model.clear_caches()
        model.offload()
        from .flashvsr_arch.models.utils import clean_vram
        clean_vram()

        # Handle crossfade blending with previous segment's tail
        if segment_index > 0 and overlap_frames > 0 and hasattr(model, '_overlap_tail'):
            prev_tail = model._overlap_tail  # [blend_frames, H, W, C] on CPU

            # The overlap region in result: first overlap_frames of the upscaled output
            # Within overlap: first (overlap - blend) frames are warmup (discard)
            # last blend_frames frames: crossfade with prev_tail
            warmup = overlap_frames - blend_frames

            if blend_frames > 0 and prev_tail is not None:
                # Linear alpha ramp for crossfade
                alpha = torch.linspace(0, 1, blend_frames).view(-1, 1, 1, 1)
                blended = (1.0 - alpha) * prev_tail + alpha * result[warmup:warmup + blend_frames]
                result = torch.cat([blended, result[overlap_frames:]], dim=0)
            else:
                result = result[overlap_frames:]
        elif segment_index > 0 and overlap_frames > 0:
            # No previous tail stored, just skip overlap
            result = result[overlap_frames:]

        # Store tail frames for next segment's crossfade
        if overlap_frames > 0 and blend_frames > 0 and result.shape[0] > blend_frames:
            model._overlap_tail = result[-blend_frames:].cpu().to(torch.float16)
        else:
            model._overlap_tail = None

        return (result, model)
