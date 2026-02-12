import subprocess
import sys
import os


def get_cupy_package():
    """Detect PyTorch's CUDA version and return the matching cupy package name."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[BIM-VFI] WARNING: CUDA not available. cupy requires CUDA.")
            return None
        cuda_version = torch.version.cuda
        if cuda_version is None:
            print("[BIM-VFI] WARNING: PyTorch has no CUDA version info.")
            return None
        major = cuda_version.split(".")[0]
        major = int(major)
        cupy_pkg = f"cupy-cuda{major}x"
        print(f"[BIM-VFI] Detected CUDA {cuda_version}, will use {cupy_pkg}")
        return cupy_pkg
    except Exception as e:
        print(f"[BIM-VFI] WARNING: Could not detect CUDA version: {e}")
        return None


def install():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ])

    # Install cupy matching the current CUDA version
    try:
        import cupy
        print("[BIM-VFI] cupy already installed, skipping.")
    except ImportError:
        cupy_pkg = get_cupy_package()
        if cupy_pkg:
            print(f"[BIM-VFI] Installing {cupy_pkg} to match PyTorch CUDA...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", cupy_pkg
            ])


if __name__ == "__main__":
    install()
