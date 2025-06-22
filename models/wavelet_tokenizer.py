import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class EMAVQ(nn.Module):
    """EMA vector-quantiser using amplitude/cos/sin representation."""

    def __init__(self, vocab_size: int, code_dim: int = 3, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, code_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.register_buffer("ema_cluster_size", torch.zeros(vocab_size))
        self.register_buffer("ema_weight", torch.zeros(vocab_size, code_dim))

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input features.

        Args:
            feats: (B, L, D)
        Returns:
            quantized features, indices, vq loss
        """
        B, L, D = feats.shape
        flat = feats.reshape(-1, D)      # safe for non-contiguous
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        quant = self.embedding(idx).reshape(B, L, D)

        if self.training:
            one_hot = F.one_hot(idx, self.vocab_size).type(flat.dtype)
            self.ema_cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
            embed = self.ema_weight / cluster_size.unsqueeze(1)
            with torch.no_grad():
                self.embedding.weight.copy_(embed)

        loss = F.mse_loss(quant, feats.detach()) + 0.25 * F.mse_loss(feats, quant.detach())
        quant = feats + (quant - feats).detach()
        return quant, idx.reshape(B, L), loss


class WaveletTokenizer(nn.Module):
    """Multi-scale wavelet tokenizer using amplitude/cos/sin VQ."""

    def __init__(self, levels: int = 3, vocab_size: int = 4096, patch_nums: Tuple[int, ...] = None):
        super().__init__()
        self.levels = levels
        self.vq = EMAVQ(vocab_size, 3)   # amp + cosφ + sinφ
        self.embedding = self.vq.embedding
        self.dtcwt = DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
        self.itcwt = DTCWTInverse(biort="near_sym_b", qshift="qshift_b")
        self.vocab_size = vocab_size
        self.Cvae = 3  # amp, cosφ, sinφ
        if patch_nums is None:
            raise ValueError(
                "WaveletTokenizer requires `patch_nums` (provide via --wpatch)."
            )
        if len(patch_nums) != levels + 1:
            raise ValueError(
                f"patch_nums length {len(patch_nums)} must equal levels+1={levels+1}"
            )
        self.patch_nums = patch_nums
        self.register_buffer('amp_mu', torch.zeros(levels))
        self.register_buffer('amp_std', torch.ones(levels))
        self.amp_scale = nn.Parameter(torch.ones(levels))
        class _Proxy:
            def __init__(self, parent: 'WaveletTokenizer'):
                self._parent = parent
                self.embedding = parent.embedding

            def idxBl_to_var_input(self, gt_ms_idx_Bl):
                return self._parent.idxBl_to_var_input(gt_ms_idx_Bl)

            def get_next_autoregressive_input(self, si, SN, f_hat, h_BChw):
                return self._parent.get_next_autoregressive_input(si, SN, f_hat, h_BChw)

            def __call__(self, x):
                return self._parent.vq(x)

        self.quantize = _Proxy(self)  # lightweight proxy for trainer compatibility

    # ---------- new helpers ----------
    @staticmethod
    def _complex_to_acs(c: torch.Tensor) -> torch.Tensor:
        """(real, imag) → (amp, cosφ, sinφ)"""
        real, imag = c.unbind(-1)
        amp = torch.sqrt(real ** 2 + imag ** 2)
        denom = amp + 1e-8
        cos = real / denom
        sin = imag / denom
        return torch.stack((amp, cos, sin), dim=-1)

    @staticmethod
    def _acs_to_complex(acs: torch.Tensor) -> torch.Tensor:
        """(amp, cosφ, sinφ) → (real, imag)"""
        amp, cos, sin = acs.unbind(-1)
        real = amp * cos
        imag = amp * sin
        return torch.stack((real, imag), dim=-1)

    def compute_token_lens(self, img_size: int, subsample: int = 1) -> List[int]:
        """Infer per-level token lengths for a given image size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=self.embedding.weight.device)
            yl, yh = self.dtcwt(dummy)
            lens = []
            for h in yh:
                B, C, O, H, W, _ = h.shape
                lens.append((C * O * H * W) // subsample)
            B, C, H, W = yl.shape
            lens.append((C * H * W) // subsample)
        return lens

    def img_to_idxBl(self, img: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]]]:
        """Decompose image and quantize to token indices."""
        yl, yh = self.dtcwt(img)
        shapes_local: List[Tuple[int, int, int, int]] = []
        idx_ls: List[torch.Tensor] = []
        for li, h in enumerate(yh):
            B, C, O, H, W, _ = h.shape
            shapes_local.append((C, O, H, W))
            acs = self._complex_to_acs(h)
            amp = acs[..., 0]
            mu = amp.mean()
            std = amp.std() + 1e-8
            if self.training:
                self.amp_mu[li] = self.amp_mu[li] * 0.99 + mu * 0.01
                self.amp_std[li] = self.amp_std[li] * 0.99 + std * 0.01
            else:
                mu, std = self.amp_mu[li], self.amp_std[li]
            acs[..., 0] = (amp - mu) / std * self.amp_scale[li]
            q = acs.permute(0, 3, 4, 2, 1, 5).reshape(B, -1, 3)
            _, idx, _ = self.vq(q)
            idx_ls.append(idx)
        B, C, H, W = yl.shape
        shapes_local.append((C, 1, H, W))
        low_c = torch.stack((yl, torch.zeros_like(yl)), dim=-1)
        low = self._complex_to_acs(low_c).unsqueeze(2)
        amp = low[..., 0]
        mu = amp.mean()
        std = amp.std() + 1e-8
        li = len(yh)
        if self.training:
            self.amp_mu[li] = self.amp_mu[li] * 0.99 + mu * 0.01
            self.amp_std[li] = self.amp_std[li] * 0.99 + std * 0.01
        else:
            mu, std = self.amp_mu[li], self.amp_std[li]
        low[..., 0] = (amp - mu) / std * self.amp_scale[li]
        _, idx, _ = self.vq(low.permute(0, 3, 4, 2, 1, 5).reshape(B, -1, 3))
        idx_ls.append(idx)
        return idx_ls, shapes_local

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], shapes) -> torch.Tensor:
        """Decode tokens back to image."""
        yh = []
        for i, idx in enumerate(ms_idx_Bl[:-1]):
            C, O, H, W = shapes[i]
            B = idx.shape[0]
            ap = (
                self.embedding(idx)
                .reshape(B, H, W, O, C, 3)
                .permute(0, 4, 3, 1, 2, 5)
            )
            amp = ap[..., 0]
            amp = amp / self.amp_scale[i]
            amp = amp * self.amp_std[i] + self.amp_mu[i]
            ap = torch.stack((amp, ap[..., 1], ap[..., 2]), dim=-1)
            c = self._acs_to_complex(ap)
            yh.append(c)
        C, _, H, W = shapes[-1]
        B = ms_idx_Bl[-1].shape[0]
        ap_low = (
            self.embedding(ms_idx_Bl[-1])
            .reshape(B, H, W, 1, C, 3)
            .permute(0, 4, 3, 1, 2, 5)
        )
        amp = ap_low[..., 0]
        amp = amp / self.amp_scale[len(shapes)-1]
        amp = amp * self.amp_std[len(shapes)-1] + self.amp_mu[len(shapes)-1]
        ap_low = torch.stack((amp, ap_low[...,1], ap_low[...,2]), dim=-1)
        yl = self._acs_to_complex(ap_low)[..., 0]
        yl = yl.squeeze(2)
        img = self.itcwt((yl, yh))
        return img.clamp(-1, 1)

    # ===== functions used by VAR trainer =====
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.patch_nums is None:
            feats = [self.embedding(idx) for idx in gt_ms_idx_Bl[:-1]]
        else:
            for idx, pn in zip(gt_ms_idx_Bl[:-1], self.patch_nums[:-1]):
                f = self.embedding(idx)
                if f.shape[1] != pn * pn:
                    f = F.interpolate(f.transpose(1, 2), size=pn * pn, mode='linear', align_corners=False).transpose(1, 2)
                feats.append(f)
        return torch.cat(feats, dim=1) if feats else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        if f_hat is None:
            return None, h_BChw
        HW = f_hat.shape[-1]
        if si != SN - 1:
            up = F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')
            f_hat.add_(up)
            return f_hat, F.interpolate(f_hat, size=(self.patch_nums[si+1], self.patch_nums[si+1]), mode='area')
        else:
            f_hat.add_(F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
            return f_hat, f_hat
