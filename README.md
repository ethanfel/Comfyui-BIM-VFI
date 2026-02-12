# ComfyUI BIM-VFI

ComfyUI custom nodes for video frame interpolation using [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) (CVPR 2025). Designed for long videos with thousands of frames — processes them without running out of VRAM.

## Nodes

### Load BIM-VFI Model

Loads the BiM-VFI checkpoint. Auto-downloads from Google Drive on first use to `ComfyUI/models/bim-vfi/`.

| Input | Description |
|-------|-------------|
| **model_path** | Checkpoint file from `models/bim-vfi/` |
| **auto_pyr_level** | Auto-select pyramid level by resolution (&lt;540p=3, 540p=5, 1080p=6, 4K=7) |
| **pyr_level** | Manual pyramid level (3-7), only used when auto is off |

### BIM-VFI Interpolate

Interpolates frames from an image batch.

| Input | Description |
|-------|-------------|
| **images** | Input image batch |
| **model** | Model from the loader node |
| **multiplier** | 2x, 4x, or 8x frame rate (recursive 2x passes) |
| **batch_size** | Frame pairs processed simultaneously (higher = faster, more VRAM) |
| **chunk_size** | Process in segments of N input frames (0 = disabled). Bounds memory for very long videos. Result is identical to processing all at once |
| **keep_device** | Keep model on GPU between pairs (faster, ~200MB constant VRAM) |
| **all_on_gpu** | Keep all intermediate frames on GPU (fast, needs large VRAM) |
| **clear_cache_after_n_frames** | Clear CUDA cache every N pairs to prevent VRAM buildup |

**Output frame count:** 2x = 2N-1, 4x = 4N-3, 8x = 8N-7

## Installation

Clone into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-user/Comfyui-BIM-VFI.git
```

Dependencies (`gdown`, `cupy`) are auto-installed on first load. The correct `cupy` variant is detected from your PyTorch CUDA version.

> **Warning:** `cupy` is a large package (~800MB) and compilation/installation can take several minutes. The first ComfyUI startup after installing this node may appear to hang while `cupy` installs in the background. Check the console log for progress. If auto-install fails (e.g. missing build tools in Docker), install manually with:
> ```bash
> pip install cupy-cuda12x  # replace 12 with your CUDA major version
> ```

To install manually:

```bash
cd Comfyui-BIM-VFI
python install.py
```

### Requirements

- PyTorch with CUDA
- `cupy` (matching your CUDA version)
- `gdown` (for model auto-download)

## VRAM Guide

| VRAM | Recommended settings |
|------|---------------------|
| 8 GB | batch_size=1, chunk_size=500 |
| 24 GB | batch_size=2-4, chunk_size=1000 |
| 48 GB+ | batch_size=4-16, all_on_gpu=true |
| 96 GB+ | batch_size=8-16, all_on_gpu=true, chunk_size=0 |

## Acknowledgments

This project wraps the official [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) implementation by the [KAIST VIC Lab](https://github.com/KAIST-VICLab). The model architecture files in `bim_vfi_arch/` are vendored from their repository with minimal modifications (relative imports, inference-only paths).

**Paper:**
> Wonyong Seo, Jihyong Oh, and Munchurl Kim.
> "BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions."
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.
> [[arXiv]](https://arxiv.org/abs/2412.11365) [[Project Page]](https://kaist-viclab.github.io/BiM-VFI_site/) [[GitHub]](https://github.com/KAIST-VICLab/BiM-VFI)

```bibtex
@inproceedings{seo2025bimvfi,
  title={BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions},
  author={Seo, Wonyong and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## License

The BiM-VFI model weights and architecture code are provided by KAIST VIC Lab for **research and education purposes only**. Commercial use requires permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr). See the [original repository](https://github.com/KAIST-VICLab/BiM-VFI) for details.
