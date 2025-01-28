# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm

from .utils import (
    Block,
    PatchEmbed,
    get_2d_sincos_pos_embed,
    trunc_normal_,
    apply_masks,
)

from .sem import SEM


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_dim=None,
        use_spectral_norm=False,
        use_sem=False,
        sem_config=None,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_spectral_norm=use_spectral_norm,
        )
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
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
        self.norm = norm_layer(embed_dim)

        if out_dim is not None:
            cls_head = nn.Linear(embed_dim, out_dim)
            if use_spectral_norm:
                cls_head = spectral_norm(cls_head)
            self.cls_head = cls_head
        else:
            self.cls_head = None

        if use_sem:
            self.sem = SEM(**sem_config)
        else:
            self.sem = None

        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_enc=None):
        if masks_enc is not None:
            if not isinstance(masks_enc, list):
                masks_enc = [masks_enc]

        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # -- mask x
        if masks_enc is not None:
            x = apply_masks(x, masks_enc)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.sem is not None:
            x = self.sem(x)

        if self.cls_head is not None:
            x = x.mean(dim=1)  # Global average pooling
            x = self.cls_head(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


# def vit_predictor(**kwargs):
#     model = VisionTransformerPredictor(
#         mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
#     )
#     return model


# def vit_tiny(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=192,
#         depth=12,
#         num_heads=3,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# def vit_small(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=384,
#         depth=12,
#         num_heads=6,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# def vit_base(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# def vit_large(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# def vit_huge(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1280,
#         depth=32,
#         num_heads=16,
#         mlp_ratio=4,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# def vit_giant(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size,
#         embed_dim=1408,
#         depth=40,
#         num_heads=16,
#         mlp_ratio=48 / 11,
#         qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model


# VIT_EMBED_DIMS = {
#     "vit_tiny": 192,
#     "vit_small": 384,
#     "vit_base": 768,
#     "vit_large": 1024,
#     "vit_huge": 1280,
#     "vit_giant": 1408,
# }
