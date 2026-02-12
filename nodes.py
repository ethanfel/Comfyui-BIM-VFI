import os
import logging
import torch
import folder_paths
from comfy.utils import ProgressBar

from .inference import BiMVFIModel
from .bim_vfi_arch import clear_backwarp_cache

logger = logging.getLogger("BIM-VFI")

# Google Drive file ID for the pretrained model
GDRIVE_FILE_ID = "18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC"
MODEL_FILENAME = "bim_vfi.pth"

# Register the model folder with ComfyUI
MODEL_DIR = os.path.join(folder_paths.models_dir, "bim-vfi")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)


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
                    "tooltip": "Process input frames in chunks of this size (0=disabled). Each chunk runs all interpolation passes independently then results are stitched seamlessly. Use for very long videos (1000+ frames) to bound memory. Result is identical to processing all at once.",
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
