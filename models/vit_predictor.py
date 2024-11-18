import torch
import math
from torch.nn.utils import spectral_norm
from torch import nn
from .utils import (
    Block,
    CrossBlock,
    get_2d_sincos_pos_embed,
    trunc_normal_,
    apply_masks,
    repeat_interleave_batch,
)


class VisionTransformerPredictor(nn.Module):
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
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__()
        predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if use_spectral_norm:
            predictor_embed = spectral_norm(predictor_embed)
        self.predictor_embed = predictor_embed

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
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
                Block(
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

        predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        if use_spectral_norm:
            predictor_proj = spectral_norm(predictor_proj)
        self.predictor_proj = predictor_proj

        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
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

    def forward(self, x, masks_enc, masks_pred):
        assert (masks_pred is not None) and (
            masks_enc is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]

        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # -- Batch Size
        B = len(x) // len(masks_enc)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_enc)

        _, N_ctxt, _ = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_pred)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_enc))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks_pred), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


class VisionTransformerCrossPredictor(nn.Module):
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
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__()
        predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if use_spectral_norm:
            predictor_embed = spectral_norm(predictor_embed)
        self.predictor_embed = predictor_embed

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
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

        predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        if use_spectral_norm:
            predictor_proj = spectral_norm(predictor_proj)
        self.predictor_proj = predictor_proj

        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
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

    def forward(self, x, masks_enc, masks_pred):
        assert (masks_pred is not None) and (
            masks_enc is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]

        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # -- Batch Size
        B = len(x) // len(masks_enc)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_enc)

        # -- prepare mask tokens
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_pred)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_enc))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks_pred), 1, 1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            pred_tokens = blk(x, pred_tokens)
        pred_tokens = self.predictor_norm(pred_tokens)

        # -- return preds for mask tokens
        pred_tokens = self.predictor_proj(pred_tokens)

        return pred_tokens
