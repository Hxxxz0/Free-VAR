import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


class EMAVQ(nn.Module):
    """Simple EMA vector quantizer operating on quaternion tokens."""

    def __init__(self, vocab_size: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(vocab_size, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / vocab_size, 1.0 / vocab_size)
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
        flat = feats.view(-1, D)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        quant = self.embedding(idx).view(B, L, D)

        if self.training:
            one_hot = F.one_hot(idx, self.vocab_size).type(flat.dtype)
            self.ema_cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(one_hot.t() @ flat, alpha=1 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.eps) / (n + self.vocab_size * self.eps) * n
            embed = self.ema_weight / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed)

        loss = F.mse_loss(quant.detach(), feats)
        quant = feats + (quant - feats).detach()
        return quant, idx.view(B, L), loss


class WaveletTokenizer(nn.Module):
    """Multi-scale wavelet tokenizer using quaternion VQ."""

    def __init__(self, levels: int = 3, vocab_size: int = 4096):
        super().__init__()
        self.levels = levels
        self.vq = EMAVQ(vocab_size, 4)
        self.embedding = self.vq.embedding
        self.dtcwt = DTCWTForward(J=levels, biort="near_sym_b", qshift="qshift_b")
        self.itcwt = DTCWTInverse(biort="near_sym_b", qshift="qshift_b")
        self.vocab_size = vocab_size
        self.Cvae = 4  # quaternion dimension
        class _Proxy:
            def __init__(self, parent: 'WaveletTokenizer'):
                self._parent = parent
                self.embedding = parent.embedding

            def idxBl_to_var_input(self, gt_ms_idx_Bl):
                return self._parent.idxBl_to_var_input(gt_ms_idx_Bl)

            def get_next_autoregressive_input(self, si, SN, f_hat, h_BChw):
                return self._parent.get_next_autoregressive_input(si, SN, f_hat, h_BChw)

        self.quantize = _Proxy(self)  # lightweight proxy for trainer compatibility
        self.shapes: List[Tuple[int, int, int, int]] = []

    @staticmethod
    def _complex_to_quaternion(c: torch.Tensor) -> torch.Tensor:
        real, imag = c[..., 0], c[..., 1]
        amp = (real.pow(2) + imag.pow(2)).sqrt()
        phase = torch.atan2(imag, real)
        return torch.stack((real, imag, amp, phase), dim=-1)

    @staticmethod
    def _quaternion_to_complex(q: torch.Tensor) -> torch.Tensor:
        real = q[..., 0]
        imag = q[..., 1]
        return torch.stack((real, imag), dim=-1)

    def img_to_idxBl(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Decompose image and quantize to token indices."""
        yl, yh = self.dtcwt(img)
        self.shapes = []
        idx_ls: List[torch.Tensor] = []
        for h in yh:
            B, C, O, H, W, _ = h.shape
            self.shapes.append((C, O, H, W))
            q = (
                self._complex_to_quaternion(h)
                .permute(0, 3, 4, 2, 1, 5)  # B H W O C 4
                .reshape(B, -1, 4)
            )
            _, idx, _ = self.vq(q)
            idx_ls.append(idx)
        B, C, H, W = yl.shape
        self.shapes.append((C, 1, H, W))
        amp = yl.abs()
        low = torch.stack(
            (yl, torch.zeros_like(yl), amp, torch.zeros_like(yl)),
            dim=-1,
        )  # B C H W 4
        _, idx, _ = self.vq(low.permute(0, 2, 3, 1, 4).reshape(B, -1, 4))
        idx_ls.append(idx)
        return idx_ls

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """Decode tokens back to image."""
        yh = []
        for i, idx in enumerate(ms_idx_Bl[:-1]):
            C, O, H, W = self.shapes[i]
            q = (
                self.embedding(idx)
                .view(idx.shape[0], H, W, O, C, 4)  # B H W O C 4
                .permute(0, 4, 3, 1, 2, 5)
            )
            c = self._quaternion_to_complex(q)
            yh.append(c)
        C, _, H, W = self.shapes[-1]
        q_low = (
            self.embedding(ms_idx_Bl[-1])
            .view(ms_idx_Bl[-1].shape[0], H, W, C, 4)
            .permute(0, 3, 1, 2, 4)
        )
        yl = q_low[..., 0]
        img = self.itcwt((yl, yh))
        return img.clamp(-1, 1)

    # ===== functions used by VAR trainer =====
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        feats = [self.embedding(idx) for idx in gt_ms_idx_Bl[:-1]]
        return torch.cat(feats, dim=1) if feats else None

    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor):
        return None, h_BChw
