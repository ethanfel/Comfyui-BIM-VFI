# ComfyUI BIM-VFI + EMA-VFI + SGM-VFI + GIMM-VFI

ComfyUI custom nodes for video frame interpolation using [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) (CVPR 2025), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) (CVPR 2023), [SGM-VFI](https://github.com/MCG-NJU/SGM-VFI) (CVPR 2024), and [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) (NeurIPS 2024). Designed for long videos with thousands of frames — processes them without running out of VRAM.

## Which model should I use?

| | BIM-VFI | EMA-VFI | SGM-VFI | GIMM-VFI |
|---|---------|---------|---------|----------|
| **Best for** | General-purpose, non-uniform motion | Fast inference, light VRAM | Large motion, occlusion-heavy scenes | High multipliers (4x/8x) in a single pass |
| **Quality** | Highest overall | Good | Best on large motion | Good |
| **Speed** | Moderate | Fastest | Slowest | Fast for 4x/8x (single pass) |
| **VRAM** | ~2 GB/pair | ~1.5 GB/pair | ~3 GB/pair | ~2.5 GB/pair |
| **Params** | ~17M | ~14–65M | ~15M + GMFlow | ~80M (RAFT) / ~123M (FlowFormer) |
| **Arbitrary timestep** | Yes | Yes (with `_t` checkpoint) | No (fixed 0.5) | Yes (native single-pass) |
| **4x/8x mode** | Recursive 2x passes | Recursive 2x passes | Recursive 2x passes | Single forward pass (or recursive) |
| **Paper** | CVPR 2025 | CVPR 2023 | CVPR 2024 | NeurIPS 2024 |
| **License** | Research only | Apache 2.0 | Apache 2.0 | Apache 2.0 |

**TL;DR:** Start with **BIM-VFI** for best quality. Use **EMA-VFI** if you need speed or lower VRAM. Use **SGM-VFI** if your video has large camera motion or fast-moving objects that the others struggle with. Use **GIMM-VFI** when you want 4x or 8x interpolation without recursive passes — it generates all intermediate frames in a single forward pass per pair.

## Nodes

### BIM-VFI

#### Load BIM-VFI Model

Loads the BiM-VFI checkpoint. Auto-downloads from Google Drive on first use to `ComfyUI/models/bim-vfi/`.

| Input | Description |
|-------|-------------|
| **model_path** | Checkpoint file from `models/bim-vfi/` |
| **auto_pyr_level** | Auto-select pyramid level by resolution (&lt;540p=3, 540p=5, 1080p=6, 4K=7) |
| **pyr_level** | Manual pyramid level (3-7), only used when auto is off |

#### BIM-VFI Interpolate

Interpolates frames from an image batch.

| Input | Description |
|-------|-------------|
| **images** | Input image batch |
| **model** | Model from the loader node |
| **multiplier** | 2x, 4x, or 8x frame rate (recursive 2x passes) |
| **batch_size** | Frame pairs processed simultaneously (higher = faster, more VRAM) |
| **chunk_size** | Process in segments of N input frames (0 = disabled). Bounds VRAM for very long videos. Result is identical to processing all at once |
| **keep_device** | Keep model on GPU between pairs (faster, ~200MB constant VRAM) |
| **all_on_gpu** | Keep all intermediate frames on GPU (fast, needs large VRAM) |
| **clear_cache_after_n_frames** | Clear CUDA cache every N pairs to prevent VRAM buildup |

#### BIM-VFI Segment Interpolate

Same as Interpolate but processes a single segment of the input. Chain multiple instances with Save nodes between them to bound peak RAM. The model pass-through output forces sequential execution.

### Tween Concat Videos

Concatenates segment video files into a single video using ffmpeg. Connect from any Segment Interpolate's model output to ensure it runs after all segments are saved. Works with all three models.

### EMA-VFI

#### Load EMA-VFI Model

Loads an EMA-VFI checkpoint. Auto-downloads from Google Drive on first use to `ComfyUI/models/ema-vfi/`. Variant (large/small) and timestep support are auto-detected from the filename.

| Input | Description |
|-------|-------------|
| **model_path** | Checkpoint file from `models/ema-vfi/` |
| **tta** | Test-time augmentation: flip input and average with unflipped result (~2x slower, slightly better quality) |

Available checkpoints:
| Checkpoint | Variant | Params | Arbitrary timestep |
|-----------|---------|--------|-------------------|
| `ours_t.pkl` | Large | ~65M | Yes |
| `ours.pkl` | Large | ~65M | No (fixed 0.5) |
| `ours_small_t.pkl` | Small | ~14M | Yes |
| `ours_small.pkl` | Small | ~14M | No (fixed 0.5) |

#### EMA-VFI Interpolate

Interpolates frames from an image batch. Same controls as BIM-VFI Interpolate.

#### EMA-VFI Segment Interpolate

Same as EMA-VFI Interpolate but processes a single segment. Same pattern as BIM-VFI Segment Interpolate.

### SGM-VFI

#### Load SGM-VFI Model

Loads an SGM-VFI checkpoint. Auto-downloads from Google Drive on first use to `ComfyUI/models/sgm-vfi/`. Variant (base/small) is auto-detected from the filename (default is small).

| Input | Description |
|-------|-------------|
| **model_path** | Checkpoint file from `models/sgm-vfi/` |
| **tta** | Test-time augmentation: flip input and average with unflipped result (~2x slower, slightly better quality) |
| **num_key_points** | Sparsity of global matching (0.0 = global everywhere, 0.5 = default balance, higher = faster) |

Available checkpoints:
| Checkpoint | Variant | Params |
|-----------|---------|--------|
| `ours-1-2-points.pkl` | Small | ~15M + GMFlow |

#### SGM-VFI Interpolate

Interpolates frames from an image batch. Same controls as BIM-VFI Interpolate.

#### SGM-VFI Segment Interpolate

Same as SGM-VFI Interpolate but processes a single segment. Same pattern as BIM-VFI Segment Interpolate.

### GIMM-VFI

#### Load GIMM-VFI Model

Loads a GIMM-VFI checkpoint. Auto-downloads from [HuggingFace](https://huggingface.co/Kijai/GIMM-VFI_safetensors) on first use to `ComfyUI/models/gimm-vfi/`. The matching flow estimator (RAFT or FlowFormer) is auto-detected and downloaded alongside the main model.

| Input | Description |
|-------|-------------|
| **model_path** | Checkpoint file from `models/gimm-vfi/` |
| **ds_factor** | Downscale factor for internal processing (1.0 = full res, 0.5 = half). Lower = less VRAM, faster, less quality. Try 0.5 for 4K inputs |

Available checkpoints:
| Checkpoint | Variant | Params | Flow estimator (auto-downloaded) |
|-----------|---------|--------|----------------------------------|
| `gimmvfi_r_arb_lpips_fp32.safetensors` | RAFT | ~80M | `raft-things_fp32.safetensors` |
| `gimmvfi_f_arb_lpips_fp32.safetensors` | FlowFormer | ~123M | `flowformer_sintel_fp32.safetensors` |

#### GIMM-VFI Interpolate

Interpolates frames from an image batch. Same controls as BIM-VFI Interpolate, plus:

| Input | Description |
|-------|-------------|
| **single_pass** | When enabled (default), generates all intermediate frames per pair in one forward pass using GIMM-VFI's arbitrary-timestep capability. No recursive 2x passes needed for 4x or 8x. Disable to use the standard recursive approach (same as BIM/EMA/SGM) |

#### GIMM-VFI Segment Interpolate

Same as GIMM-VFI Interpolate but processes a single segment. Same pattern as BIM-VFI Segment Interpolate.

**Output frame count (all models):** 2x = 2N-1, 4x = 4N-3, 8x = 8N-7

## Installation

Clone into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-user/ComfyUI-Tween.git
```

Dependencies (`gdown`, `cupy`, `timm`, `omegaconf`, `easydict`, `yacs`, `einops`, `huggingface_hub`) are auto-installed on first load. The correct `cupy` variant is detected from your PyTorch CUDA version.

> **Warning:** `cupy` is a large package (~800MB) and compilation/installation can take several minutes. The first ComfyUI startup after installing this node may appear to hang while `cupy` installs in the background. Check the console log for progress. If auto-install fails (e.g. missing build tools in Docker), install manually with:
> ```bash
> pip install cupy-cuda12x  # replace 12 with your CUDA major version
> ```

To install manually:

```bash
cd ComfyUI-Tween
python install.py
```

### Requirements

- PyTorch with CUDA
- `cupy` (matching your CUDA version, for BIM-VFI, SGM-VFI, and GIMM-VFI)
- `timm` (for EMA-VFI and SGM-VFI)
- `gdown` (for BIM-VFI/EMA-VFI/SGM-VFI model auto-download)
- `omegaconf`, `easydict`, `yacs`, `einops` (for GIMM-VFI)
- `huggingface_hub` (for GIMM-VFI model auto-download)

## VRAM Guide

| VRAM | Recommended settings |
|------|---------------------|
| 8 GB | batch_size=1, chunk_size=500 |
| 24 GB | batch_size=2-4, chunk_size=1000 |
| 48 GB+ | batch_size=4-16, all_on_gpu=true |
| 96 GB+ | batch_size=8-16, all_on_gpu=true, chunk_size=0 |

## Acknowledgments

This project wraps the official [BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) implementation by the [KAIST VIC Lab](https://github.com/KAIST-VICLab), the official [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) implementation by MCG-NJU, the official [SGM-VFI](https://github.com/MCG-NJU/SGM-VFI) implementation by MCG-NJU, and the [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) implementation by S-Lab (NTU). GIMM-VFI architecture files in `gimm_vfi_arch/` are adapted from [kijai/ComfyUI-GIMM-VFI](https://github.com/kijai/ComfyUI-GIMM-VFI) with safetensors checkpoints from [Kijai/GIMM-VFI_safetensors](https://huggingface.co/Kijai/GIMM-VFI_safetensors). Architecture files in `bim_vfi_arch/`, `ema_vfi_arch/`, `sgm_vfi_arch/`, and `gimm_vfi_arch/` are vendored from their respective repositories with minimal modifications (relative imports, device-awareness fixes, inference-only paths).

**BiM-VFI:**
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

**EMA-VFI:**
> Guozhen Zhang, Yuhan Zhu, Haonan Wang, Youxin Chen, Gangshan Wu, and Limin Wang.
> "Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation."
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
> [[arXiv]](https://arxiv.org/abs/2303.00440) [[GitHub]](https://github.com/MCG-NJU/EMA-VFI)

```bibtex
@inproceedings{zhang2023emavfi,
  title={Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation},
  author={Zhang, Guozhen and Zhu, Yuhan and Wang, Haonan and Chen, Youxin and Wu, Gangshan and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

**SGM-VFI:**
> Guozhen Zhang, Yuhan Zhu, Evan Zheran Liu, Haonan Wang, Mingzhen Sun, Gangshan Wu, and Limin Wang.
> "Sparse Global Matching for Video Frame Interpolation with Large Motion."
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024.
> [[arXiv]](https://arxiv.org/abs/2404.06913) [[GitHub]](https://github.com/MCG-NJU/SGM-VFI)

```bibtex
@inproceedings{zhang2024sgmvfi,
  title={Sparse Global Matching for Video Frame Interpolation with Large Motion},
  author={Zhang, Guozhen and Zhu, Yuhan and Liu, Evan Zheran and Wang, Haonan and Sun, Mingzhen and Wu, Gangshan and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

**GIMM-VFI:**
> Zujin Guo, Wei Li, and Chen Change Loy.
> "Generalizable Implicit Motion Modeling for Video Frame Interpolation."
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.
> [[arXiv]](https://arxiv.org/abs/2407.08680) [[GitHub]](https://github.com/GSeanCDAT/GIMM-VFI)

```bibtex
@inproceedings{guo2024gimmvfi,
  title={Generalizable Implicit Motion Modeling for Video Frame Interpolation},
  author={Guo, Zujin and Li, Wei and Loy, Chen Change},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## License

The BiM-VFI model weights and architecture code are provided by KAIST VIC Lab for **research and education purposes only**. Commercial use requires permission from the principal investigator (Prof. Munchurl Kim, mkimee@kaist.ac.kr). See the [original repository](https://github.com/KAIST-VICLab/BiM-VFI) for details.

The EMA-VFI model weights and architecture code are released under the [Apache 2.0 License](https://github.com/MCG-NJU/EMA-VFI/blob/main/LICENSE). See the [original repository](https://github.com/MCG-NJU/EMA-VFI) for details.

The SGM-VFI model weights and architecture code are released under the [Apache 2.0 License](https://github.com/MCG-NJU/SGM-VFI/blob/main/LICENSE). See the [original repository](https://github.com/MCG-NJU/SGM-VFI) for details.

The GIMM-VFI model weights and architecture code are released under the [Apache 2.0 License](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE). See the [original repository](https://github.com/GSeanCDAT/GIMM-VFI) for details. ComfyUI adaptation based on [kijai/ComfyUI-GIMM-VFI](https://github.com/kijai/ComfyUI-GIMM-VFI).
