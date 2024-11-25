import torch
import torch.nn.functional as F
from torch import nn
from .utils import Projector, FullGatherLayer, off_diagonal


class VICReg(nn.Module):
    def __init__(
        self,
        projector_dims,
        coeffs,
        std_cov_grad,
    ):
        super().__init__()
        self.inv_coeff, self.std_coeff, self.cov_coeff = coeffs
        self.std_cov_grad = std_cov_grad

        self.projector = Projector(projector_dims)

    def forward(self, z_ctx, z_tgt, z_pred_tgt):
        inv_loss = F.mse_loss(z_pred_tgt, z_tgt)

        if not self.std_cov_grad:
            z_ctx = z_ctx.detach()

        num_features = z_ctx.size(-1)
        z_ctx = z_ctx.view(-1, num_features)
        z_ctx = self.projector(z_ctx)

        z_ctx = torch.cat(FullGatherLayer.apply(z_ctx), dim=0)
        z_ctx = z_ctx - z_ctx.mean(dim=0)

        std_z_ctx = torch.sqrt(z_ctx.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_z_ctx))

        gathered_batch_size = z_ctx.size(0)
        num_features = z_ctx.size(-1)
        cov_z_ctx = (z_ctx.T @ z_ctx) / (gathered_batch_size - 1)
        cov_loss = off_diagonal(cov_z_ctx).pow_(2).sum().div(num_features)

        loss = (
            self.inv_coeff * inv_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss, inv_loss, std_loss, cov_loss
