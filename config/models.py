import torch.nn as nn
from functools import partial


def vit_predictor_tiny(
    img_size=224,
    patch_size=16,
):
    return {
        "num_patches": (img_size // patch_size) ** 2,
        "embed_dim": 192,
        "predictor_embed_dim": 192,
        "depth": 4,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "init_std": 0.02,
    }


def vit_tiny(
    img_size=224,
    patch_size=16,
    out_dim=None,
):
    return {
        "img_size": [img_size],
        "patch_size": patch_size,
        "in_chans": 3,
        "embed_dim": 192,
        "predictor_embed_dim": 192,
        "depth": 12,
        "predictor_depth": 4,
        "num_heads": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "init_std": 0.02,
        "out_dim": out_dim,
    }


model_configs = {"vit_tiny": vit_tiny, "vit_predictor_tiny": vit_predictor_tiny}


def get_model_config(model_name, *args, **kwargs):
    assert model_name in model_configs, "Invalid model name."

    return model_configs[model_name](*args, **kwargs)
