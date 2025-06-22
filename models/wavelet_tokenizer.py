import math
from typing import List, Tuple, Optional

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

    def __init__(self, levels: int = 3, vocab_size: int = 4096, learn_scaling: bool = False):
        super().__init__()
        self.levels = levels
        self.vq = EMAVQ(vocab_size, 3)   # amp + cosφ + sinφ
        self.embedding = self.vq.embedding
        self.dtcwt = DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
        self.itcwt = DTCWTInverse(biort="near_sym_b", qshift="qshift_b")
        self.vocab_size = vocab_size
        self.Cvae = 3  # amp, cosφ, sinφ
        self.learn_scaling = learn_scaling
        scale = torch.ones(levels + 1)
        if learn_scaling:
            self.level_scale = nn.Parameter(scale)
            self._scale_initialized = False
        else:
            self.register_buffer('level_scale', scale)
            self._scale_initialized = True
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

    def img_to_idxBl(self, img: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, int]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Decompose image and quantize to token indices.

        Returns:
            idx_ls: list of token indices for each level
            shapes_local: shapes for each level
            amp_stats: list of (mean, std) tuples for amplitude at each level
        """
        yl, yh = self.dtcwt(img)
        shapes_local: List[Tuple[int, int, int, int]] = []
        idx_ls: List[torch.Tensor] = []
        amp_stats: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for li, h in enumerate(yh):
            B, C, O, H, W, _ = h.shape
            shapes_local.append((C, O, H, W))
            real, imag = h.unbind(-1)
            amp = torch.sqrt(real ** 2 + imag ** 2)
            mean = amp.mean(dim=(1, 2, 3, 4))
            std = amp.std(dim=(1, 2, 3, 4)) + 1e-8
            if self.learn_scaling and not self._scale_initialized:
                self.level_scale.data[li] = (1.0 / std.mean()).detach()
            amp_n = (amp - mean[:, None, None, None, None]) / std[:, None, None, None, None]
            amp_n = amp_n * self.level_scale[li]
            cos = real / (amp + 1e-8)
            sin = imag / (amp + 1e-8)
            q = (
                torch.stack((amp_n, cos, sin), dim=-1)
                .permute(0, 3, 4, 2, 1, 5)  # B H W O C 3
                .reshape(B, -1, 3)
            )
            _, idx, _ = self.vq(q)
            idx_ls.append(idx)
            amp_stats.append((mean, std))

        B, C, H, W = yl.shape
        shapes_local.append((C, 1, H, W))
        amp = yl.abs()
        mean = amp.mean(dim=(1, 2, 3))
        std = amp.std(dim=(1, 2, 3)) + 1e-8
        if self.learn_scaling and not self._scale_initialized:
            self.level_scale.data[-1] = (1.0 / std.mean()).detach()
            self._scale_initialized = True
        amp_n = (amp - mean[:, None, None, None]) / std[:, None, None, None]
        amp_n = amp_n * self.level_scale[-1]
        cos = yl / (amp + 1e-8)
        sin = torch.zeros_like(yl)
        q = torch.stack((amp_n, cos, sin), dim=-1).unsqueeze(2)
        _, idx, _ = self.vq(q.permute(0, 3, 4, 2, 1, 5).reshape(B, -1, 3))
        idx_ls.append(idx)
        amp_stats.append((mean, std))

        return idx_ls, shapes_local, amp_stats

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], shapes, amp_stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """Decode tokens back to image using stored amplitude statistics."""
        if amp_stats is None:
            amp_stats = [(torch.zeros(1, device=self.level_scale.device), torch.ones(1, device=self.level_scale.device))] * (self.levels + 1)

        yh = []
        for i, idx in enumerate(ms_idx_Bl[:-1]):
            C, O, H, W = shapes[i]
            mean, std = amp_stats[i]
            B = idx.shape[0]
            ap = (
                self.embedding(idx)
                .reshape(B, H, W, O, C, 3)
                .permute(0, 4, 3, 1, 2, 5)
        )
            amp = ap[..., 0] / self.level_scale[i]
            amp = amp * std[:, None, None, None, None] + mean[:, None, None, None, None]
            ap = torch.stack((amp, ap[..., 1], ap[..., 2]), dim=-1)
            c = self._acs_to_complex(ap)
            yh.append(c)
        C, _, H, W = shapes[-1]
        mean, std = amp_stats[-1]
        B = ms_idx_Bl[-1].shape[0]
        ap_low = (
            self.embedding(ms_idx_Bl[-1])
            .reshape(B, H, W, 1, C, 3)
            .permute(0, 4, 3, 1, 2, 5)
        )
        amp = ap_low[..., 0] / self.level_scale[-1]
        amp = amp * std[:, None, None, None] + mean[:, None, None, None]
        ap_low = torch.stack((amp, ap_low[..., 1], ap_low[..., 2]), dim=-1)
        yl = self._acs_to_complex(ap_low)[..., 0]
        yl = yl.squeeze(2)
        img = self.itcwt((yl, yh))
        return img.clamp(-1, 1)

    # ===== functions used by VAR trainer =====
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        feats = [self.embedding(idx) for idx in gt_ms_idx_Bl[:-1]]
        return torch.cat(feats, dim=1) if feats else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        return None, h_BChw
