import torch
import torch.nn.functional as F
from torch import nn
from .utils import (
    Projector,
    FullGatherLayer,
    off_diagonal,
    repeat_interleave_batch,
    apply_masks,
)


class VICReg(nn.Module):
    def __init__(
        self,
        encoder,
        target,
        predictor,
        projector_dims,
        coeffs,
        project,
        std_cov_grad,
    ):
        super().__init__()
        self.encoder = encoder
        self.target = target
        self.predictor = predictor
        self.inv_coeff, self.std_coeff, self.cov_coeff = coeffs
        self.project = project
        self.std_cov_grad = std_cov_grad

        self.encoder_projector = Projector(projector_dims) if project else None
        self.target_projector = Projector(projector_dims) if project else None

        self.target.requires_grad_(False)
        if project:
            self.target_projector.requires_grad_(False)

    def _forward_target(self, x, masks_ctx, masks_tgt):
        with torch.no_grad():
            z_tgt = self.target(x)
            z_tgt = F.layer_norm(z_tgt, (z_tgt.size(-1),))
            B = len(z_tgt)

            # -- create targets (masked regions of tgt) --
            z_tgt = apply_masks(z_tgt, masks_tgt)
            z_tgt = repeat_interleave_batch(z_tgt, B, repeat=len(masks_ctx))

            return z_tgt

    def _update_target(self, momentum):
        with torch.no_grad():
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target.parameters()
            ):
                param_k.data.mul_(momentum).add_(
                    (1.0 - momentum) * param_q.detach().data
                )

            if self.project:
                for param_q, param_k in zip(
                    self.encoder_projector.parameters(),
                    self.target_projector.parameters(),
                ):
                    param_k.data.mul_(momentum).add_(
                        (1.0 - momentum) * param_q.detach().data
                    )

    def forward(self, x, masks_ctx, masks_tgt):
        z_ctx = self.encoder(x, masks_ctx)
        z_tgt = self._forward_target(x, masks_ctx, masks_tgt)

        B_ctx, L_ctx, _ = z_ctx.shape
        B_tgt, L_tgt, _ = z_tgt.shape

        if self.project:
            # Flatten for projection
            z_ctx = z_ctx.view(-1, z_ctx.size(-1))
            z_tgt = z_tgt.view(-1, z_tgt.size(-1))

            z_ctx = self.encoder_projector(z_ctx)

            with torch.no_grad():
                z_tgt = self.target_projector(z_tgt)

            # Revert to original shape for predictor
            z_ctx = z_ctx.view(B_ctx, L_ctx, -1)
            z_tgt = z_tgt.view(B_tgt, L_tgt, -1)

        z_pred_tgt = self.predictor(z_ctx, masks_ctx, masks_tgt)

        # 1. Compute invariance loss
        inv_loss = F.mse_loss(z_pred_tgt, z_tgt)

        if not self.std_cov_grad:
            z_ctx = z_ctx.detach()

        # Flatten for loss computation
        z_ctx = z_ctx.view(-1, z_ctx.size(-1))

        # 2. Gather from all gpus
        z_ctx = torch.cat(FullGatherLayer.apply(z_ctx.contiguous()), dim=0)
        z_ctx = z_ctx - z_ctx.mean(dim=0)

        # 3. Compute variance loss
        std_z_ctx = torch.sqrt(z_ctx.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z_ctx))

        # 4. Compute covariance loss
        cov_z_ctx = (z_ctx.T @ z_ctx) / (z_ctx.size(0) - 1)
        cov_loss = off_diagonal(cov_z_ctx).pow_(2).sum().div(z_ctx.size(-1))

        loss = (
            self.inv_coeff * inv_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss, inv_loss, std_loss, cov_loss
