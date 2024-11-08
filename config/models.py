import torch.nn as nn
from functools import partial


def vit_predictor(
    img_size=224,
    patch_size=16,
):
    return {
        "num_patches": (img_size // patch_size) ** 2,
        "embed_dim": 192,
        "predictor_embed_dim": 384,
        "depth": 6,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "init_std": 0.02,
    }


def vit_tiny(
    img_size=224,
    patch_size=16,
):
    return {
        "img_size": [img_size],
        "patch_size": patch_size,
        "in_chans": 3,
        "embed_dim": 192,
        "predictor_embed_dim": 384,
        "depth": 12,
        "predictor_depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "norm_layer": nn.LayerNorm,
        "init_std": 0.02,
    }
