import torch
import torch.nn.functional as F
import torch.nn as nn

from .backwarp import backwarp
from .resnet_encoder import ResNetPyramid
from .caun import CAUN
from .bimfn import BiMFN
from .sn import SynthesisNetwork

from ..utils.padder import InputPadder


class BiMVFI(nn.Module):
    def __init__(self, pyr_level=3, feat_channels=32, **kwargs):
        super(BiMVFI, self).__init__()
        self.pyr_level = pyr_level
        self.mfe = ResNetPyramid(feat_channels)
        self.cfe = ResNetPyramid(feat_channels)
        self.bimfn = BiMFN(feat_channels)
        self.sn = SynthesisNetwork(feat_channels)
        self.feat_channels = feat_channels
        self.caun = CAUN(feat_channels)

    def forward_one_lvl(self, img0, img1, last_flow, last_occ, time_period=0.5):
        feat0_pyr = self.mfe(img0)
        feat1_pyr = self.mfe(img1)
        cfeat0_pyr = self.cfe(img0)
        cfeat1_pyr = self.cfe(img1)

        B, _, H, W = feat0_pyr[-1].shape

        # Inference path: prepare uniform BiM
        r = torch.ones((B, 1, H, W), device=feat0_pyr[-1].device) * time_period
        phi = torch.ones((B, 1, H, W), device=feat0_pyr[-1].device) * torch.pi
        phi = torch.cat([torch.cos(phi), torch.sin(phi)], dim=1)

        last_flow = F.interpolate(
            input=last_flow.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False) * 0.5
        last_occ = F.interpolate(
            input=last_occ.detach().clone(), scale_factor=0.5,
            mode="bilinear", align_corners=False)

        flow_low, flow_res = self.bimfn(
            feat0_pyr[-1], feat1_pyr[-1], r, phi, last_flow, last_occ)

        bi_flow_pyr, occ = self.caun(flow_low, cfeat0_pyr, cfeat1_pyr, last_occ)
        flow = bi_flow_pyr[0]

        interp_img, occ, extra_dict = self.sn(
            img0, img1, cfeat0_pyr, cfeat1_pyr, bi_flow_pyr, occ)
        extra_dict.update({'flow_res': flow_res})
        return flow, occ, interp_img, extra_dict

    def forward(self, img0, img1, time_step,
                pyr_level=None, **kwargs):
        if pyr_level is None: pyr_level = self.pyr_level
        N, _, H, W = img0.shape
        interp_imgs = []

        padder = InputPadder(img0.shape, divisor=int(2 ** (pyr_level + 1)))

        # Normalize input images
        with torch.set_grad_enabled(False):
            tenStats = [img0, img1]
            tenMean_ = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd_ = (sum([tenIn.std([1, 2, 3], False, True).square() + (
                    tenMean_ - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()

            img0 = (img0 - tenMean_) / (tenStd_ + 0.0000001)
            img1 = (img1 - tenMean_) / (tenStd_ + 0.0000001)

        # Pad images for downsampling
        img0, img1 = padder.pad(img0, img1)

        N, _, H, W = img0.shape

        for level in list(range(pyr_level))[::-1]:
            # Downsample images if needed
            if level != 0:
                scale_factor = 1 / 2 ** level
                img0_this_lvl = F.interpolate(
                    input=img0, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
                img1_this_lvl = F.interpolate(
                    input=img1, scale_factor=scale_factor,
                    mode="bilinear", align_corners=False, antialias=True)
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1

            # Initialize zero flows for lowest pyramid level
            if level == pyr_level - 1:
                last_flow = torch.zeros(
                    (N, 4, H // (2 ** (level + 1)), W // (2 ** (level + 1))), device=img0.device
                )
                last_occ = torch.zeros(N, 1, H // (2 ** (level + 1)), W // (2 ** (level + 1)), device=img0.device)
            else:
                last_flow = flow
                last_occ = occ

            # Single pyramid level run
            flow, occ, interp_img, extra_dict = self.forward_one_lvl(
                img0_this_lvl, img1_this_lvl, last_flow, last_occ, time_step)

            interp_imgs.append((interp_img) * (tenStd_ + 0.0000001) + tenMean_)

        result_dict = {
            "imgt_preds": interp_imgs,
            'imgt_pred': padder.unpad(interp_imgs[-1].contiguous()),
        }

        return result_dict
