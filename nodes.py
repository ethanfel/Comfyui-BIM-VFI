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
                "model_path": (get_available_models(), {"default": MODEL_FILENAME}),
                "pyr_level": ("INT", {"default": 3, "min": 3, "max": 7, "step": 1}),
            }
        }

    RETURN_TYPES = ("BIM_VFI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "video/BIM-VFI"

    def load_model(self, model_path, pyr_level):
        full_path = os.path.join(MODEL_DIR, model_path)

        if not os.path.exists(full_path):
            logger.info(f"Model not found at {full_path}, attempting download...")
            download_model_from_gdrive(GDRIVE_FILE_ID, full_path)

        wrapper = BiMVFIModel(
            checkpoint_path=full_path,
            pyr_level=pyr_level,
            device="cpu",
        )

        logger.info(f"BIM-VFI model loaded (pyr_level={pyr_level})")
        return (wrapper,)


class BIMVFIInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("BIM_VFI_MODEL",),
                "multiplier": ([2, 4, 8], {"default": 2}),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "keep_device": ("BOOLEAN", {"default": True}),
                "all_on_gpu": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/BIM-VFI"

    def interpolate(self, images, model, multiplier, clear_cache_after_n_frames, keep_device, all_on_gpu):
        if images.shape[0] < 2:
            return (images,)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_passes = {2: 1, 4: 2, 8: 3}[multiplier]

        # all_on_gpu implies keep_device
        if all_on_gpu:
            keep_device = True

        # Where to store intermediate frames
        storage_device = device if all_on_gpu else torch.device("cpu")

        # Convert from ComfyUI [B, H, W, C] to model [B, C, H, W]
        frames = images.permute(0, 3, 1, 2).to(storage_device)

        # After each 2x pass, frame count = 2*N - 1, so compute total pairs across passes
        n = frames.shape[0]
        total_steps = 0
        for _ in range(num_passes):
            total_steps += n - 1
            n = 2 * n - 1

        pbar = ProgressBar(total_steps)
        step = 0

        if keep_device:
            model.to(device)

        for pass_idx in range(num_passes):
            new_frames = []
            num_pairs = frames.shape[0] - 1

            for i in range(num_pairs):
                frame0 = frames[i:i+1]   # [1, C, H, W]
                frame1 = frames[i+1:i+2] # [1, C, H, W]

                if not keep_device:
                    model.to(device)

                mid = model.interpolate_pair(frame0, frame1, time_step=0.5)
                mid = mid.to(storage_device)

                if not keep_device:
                    model.to("cpu")

                new_frames.append(frames[i:i+1])
                new_frames.append(mid)

                step += 1
                pbar.update_absolute(step, total_steps)

                if not all_on_gpu and (i + 1) % clear_cache_after_n_frames == 0 and torch.cuda.is_available():
                    clear_backwarp_cache()
                    torch.cuda.empty_cache()

            # Append last frame
            new_frames.append(frames[-1:])
            frames = torch.cat(new_frames, dim=0)

            if not all_on_gpu and torch.cuda.is_available():
                clear_backwarp_cache()
                torch.cuda.empty_cache()

        # Convert back to ComfyUI [B, H, W, C], on CPU for ComfyUI
        result = frames.cpu().permute(0, 2, 3, 1)
        return (result,)
