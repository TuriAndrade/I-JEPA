import torch
import math
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch import nn
from .utils import (
    CrossBlock,
    get_2d_sincos_pos_embed,
    trunc_normal_,
    apply_masks,
    repeat_interleave_batch,
)


class MutualInformationEstimator(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        ma_rate=0.01,
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__()
        predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if use_spectral_norm:
            predictor_embed = spectral_norm(predictor_embed)
        self.predictor_embed = predictor_embed

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False
        )
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1], int(num_patches**0.5), cls_token=False
        )
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        )
        # --
        self.predictor_blocks = nn.ModuleList(
            [
                CrossBlock(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_spectral_norm=use_spectral_norm,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)

        self.predictor_head = (
            nn.Sequential(
                spectral_norm(
                    nn.Linear(
                        predictor_embed_dim,
                        int(mlp_ratio * predictor_embed_dim),
                        bias=True,
                    )
                ),
                nn.ReLU(),
                spectral_norm(nn.Linear(int(mlp_ratio * predictor_embed_dim), 1)),
            )
            if use_spectral_norm
            else nn.Sequential(
                nn.Linear(
                    predictor_embed_dim,
                    int(mlp_ratio * predictor_embed_dim),
                    bias=True,
                ),
                nn.ReLU(),
                nn.Linear(int(mlp_ratio * predictor_embed_dim), 1),
            )
        )

        self.ma_rate = ma_rate
        self.ma_et = None  # Initialize moving average
        # ------
        self.init_std = init_std

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def f(self, ctx, tgt, masks_enc, masks_pred):
        assert (masks_pred is not None) and (
            masks_enc is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]

        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        B = B_ctx = B_tgt = len(ctx) // len(masks_enc)
        # B_tgt = len(tgt) // len(masks_pred) = B_ctx = len(ctx) // len(masks_enc)

        # -- Map from encoder-dim to predictor-dim
        ctx = self.predictor_embed(ctx)
        tgt = self.predictor_embed(tgt)

        # -- Get positional embeddings
        x_pos_embed = self.predictor_pos_embed.repeat(
            B, 1, 1
        )  # Shape: [B, num_patches, predictor_embed_dim]

        # -- Apply masks to positional embeddings to get embeddings for ctx and tgt
        ctx_pos_embed = apply_masks(
            x_pos_embed, masks_enc
        )  # Shape: [B * len(masks_enc), N_ctx, predictor_embed_dim]
        tgt_pos_embed = apply_masks(
            x_pos_embed, masks_pred
        )  # Shape: [B * len(masks_pred), N_tgt, predictor_embed_dim]

        # -- Add positional embeddings
        ctx += ctx_pos_embed
        tgt += tgt_pos_embed

        # -- Repeat to match dimesions
        ctx = ctx.repeat(
            len(masks_pred), 1, 1
        )  # [B * len(masks_pred) * len(masks_enc), N_ctx, predictor_embed_dim]
        tgt = repeat_interleave_batch(
            tgt, B, repeat=len(masks_enc)
        )  # [B * len(masks_pred) * len(masks_enc), N_tgt, predictor_embed_dim]

        # -- Cross attn fwd prop
        x = tgt
        for blk in self.predictor_blocks:
            x = blk(ctx, x)

        # -- Return scalar estimation
        x = self.predictor_norm(x)
        x = self.predictor_head(x)  # Shape: [B, N, 1]
        x = x.mean(dim=1)  # Aggregate over sequence length to get [B, 1]

        return x.squeeze(-1)

    def forward(self, ctx, tgt, masks_enc, masks_pred):
        # Compute network outputs for joint samples
        t_joint = self.f(ctx, tgt, masks_enc, masks_pred)
        t_joint = torch.mean(t_joint)  # Shape: scalar

        # Shuffle tgt to create marginal samples
        shuffled_tgt = tgt[torch.randperm(tgt.size(0))]

        # Compute network outputs for marginal samples
        t_marginal = self.f(
            ctx, shuffled_tgt, masks_enc, masks_pred
        )  # Shape: [batch_size]

        # Compute log of mean(exp(t_marginal)) using the log-sum-exp trick
        batch_size = t_marginal.size(0)
        log_mean_exp_t_marginal = torch.logsumexp(t_marginal, dim=0) - torch.log(
            torch.tensor(batch_size, device=t_marginal.device)
        )

        # Initialize moving average in log domain if it doesn't exist
        if self.ma_et is None:
            self.ma_et = log_mean_exp_t_marginal.detach()
        else:
            # Update moving average of log_mean_exp_t_marginal in log domain
            ma_rate = torch.tensor(self.ma_rate, device=t_marginal.device)
            self.ma_et = torch.logsumexp(
                torch.stack(
                    [
                        self.ma_et + torch.log(1 - ma_rate),
                        log_mean_exp_t_marginal.detach() + torch.log(ma_rate),
                    ]
                ),
                dim=0,
            )

        # Compute the lower bound on mutual information using the updated moving average
        mi_lb = t_joint - self.ma_et

        # Total loss is a weighted sum of MSE and negative mutual information lower bound
        loss = -mi_lb

        return loss
