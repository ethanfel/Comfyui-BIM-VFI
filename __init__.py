import subprocess
import sys
import logging

logger = logging.getLogger("BIM-VFI")


def _auto_install_deps():
    """Auto-install missing dependencies on first load."""
    # gdown
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("[BIM-VFI] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

    # cupy
    try:
        import cupy  # noqa: F401
    except ImportError:
        try:
            import torch
            major = int(torch.version.cuda.split(".")[0])
            cupy_pkg = f"cupy-cuda{major}x"
            logger.info(f"[BIM-VFI] Installing {cupy_pkg} (CUDA {torch.version.cuda})...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", cupy_pkg])
        except Exception as e:
            logger.warning(f"[BIM-VFI] Could not auto-install cupy: {e}")


_auto_install_deps()

from .nodes import LoadBIMVFIModel, BIMVFIInterpolate

NODE_CLASS_MAPPINGS = {
    "LoadBIMVFIModel": LoadBIMVFIModel,
    "BIMVFIInterpolate": BIMVFIInterpolate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBIMVFIModel": "Load BIM-VFI Model",
    "BIMVFIInterpolate": "BIM-VFI Interpolate",
}
