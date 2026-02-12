import torch
from .bim_vfi_arch import BiMVFI


class BiMVFIModel:
    """Clean inference wrapper around BiMVFI for ComfyUI integration."""

    def __init__(self, checkpoint_path, pyr_level=3, auto_pyr_level=True, device="cpu"):
        self.pyr_level = pyr_level
        self.auto_pyr_level = auto_pyr_level
        self.device = device

        self.model = BiMVFI(pyr_level=pyr_level, feat_channels=32)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()
        self.model.to(device)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip common prefixes (e.g. "module." from DDP or "model." from wrapper)
        cleaned = {}
        for k, v in state_dict.items():
            key = k
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("model."):
                key = key[len("model."):]
            cleaned[key] = v

        self.model.load_state_dict(cleaned)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @torch.no_grad()
    def interpolate_pair(self, frame0, frame1, time_step=0.5):
        """Interpolate a single frame between two input frames.

        Args:
            frame0: [1, C, H, W] tensor, float32, range [0, 1]
            frame1: [1, C, H, W] tensor, float32, range [0, 1]
            time_step: float in (0, 1), temporal position of interpolated frame

        Returns:
            Interpolated frame as [1, C, H, W] tensor, float32, clamped to [0, 1]
        """
        device = next(self.model.parameters()).device
        img0 = frame0.to(device)
        img1 = frame1.to(device)

        if self.auto_pyr_level:
            _, _, h, _ = img0.shape
            if h >= 2160:
                pyr_level = 7
            elif h >= 1080:
                pyr_level = 6
            elif h >= 540:
                pyr_level = 5
            else:
                pyr_level = 3
        else:
            pyr_level = self.pyr_level

        time_step_tensor = torch.tensor([time_step], device=device).view(1, 1, 1, 1)

        result_dict = self.model(
            img0=img0, img1=img1,
            time_step=time_step_tensor,
            pyr_level=pyr_level,
        )

        interp = result_dict["imgt_pred"]
        interp = torch.clamp(interp, 0, 1)
        return interp
