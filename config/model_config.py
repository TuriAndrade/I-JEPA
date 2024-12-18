import torch.nn as nn
from functools import partial


def vit_predictor_tiny(
    img_size=224,
    patch_size=16,
    use_spectral_norm=False,
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
        "use_spectral_norm": use_spectral_norm,
    }


def mutual_information_predictor_tiny(
    img_size=224,
    patch_size=16,
    use_spectral_norm=False,
    mse_factor=1,
    mi_factor=1,
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
        "ma_rate": 0.01,
        "mse_factor": mse_factor,
        "mi_factor": mi_factor,
        "use_spectral_norm": use_spectral_norm,
    }


def mutual_information_estimator_tiny(
    img_size=224,
    patch_size=16,
    use_spectral_norm=False,
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
        "ma_rate": 0.01,
        "use_spectral_norm": use_spectral_norm,
    }


def vit_tiny(
    img_size=224,
    patch_size=16,
    out_dim=None,
    use_spectral_norm=False,
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
        "use_spectral_norm": use_spectral_norm,
    }


def vic_reg_25_25_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 1, 0.04],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_10_10_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 1, 0.1],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_25_1_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 0.04, 0.04],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_25_2_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 0.08, 0.04],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_25_4_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 0.16, 0.04],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_10_1_1_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 0.1, 0.1],
        "std_cov_grad": apply_reg,
        "project": project,
    }


def vic_reg_25_4_2_tiny(apply_reg=True, project=True):
    return {
        "projector_dims": [192, 192, 192],
        "coeffs": [1, 0.16, 0.08],
        "std_cov_grad": apply_reg,
        "project": project,
    }


model_configs = {
    "vit_tiny": vit_tiny,
    "vic_reg_25_25_1_tiny": vic_reg_25_25_1_tiny,
    "vic_reg_10_10_1_tiny": vic_reg_10_10_1_tiny,
    "vic_reg_25_1_1_tiny": vic_reg_25_1_1_tiny,
    "vic_reg_25_2_1_tiny": vic_reg_25_2_1_tiny,
    "vic_reg_25_4_1_tiny": vic_reg_25_4_1_tiny,
    "vic_reg_10_1_1_tiny": vic_reg_10_1_1_tiny,
    "vic_reg_25_4_2_tiny": vic_reg_25_4_2_tiny,
    "vit_predictor_tiny": vit_predictor_tiny,
    "mutual_information_predictor_tiny": mutual_information_predictor_tiny,
    "mutual_information_estimator_tiny": mutual_information_estimator_tiny,
}


def get_model_config(model_name, *args, **kwargs):
    assert model_name in model_configs, "Invalid model name."

    return model_configs[model_name](*args, **kwargs)
