from models import VisionTransformer, VisionTransformerPredictor
from trainers import DDPIJepaTrainer
from torch import nn
from functools import partial
from batch_collators import MBMaskCollator, norm_img, img_channels_first


def main():
    img_size = 32
    patch_size = 2

    #
    # Encoder
    #
    encoder = VisionTransformer
    encoder_config = {
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

    #
    # Predictor
    #
    predictor = VisionTransformerPredictor
    predictor_config = {
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

    #
    # Masks
    #
    batch_collator = MBMaskCollator
    batch_collator_config = {
        "input_size": (img_size, img_size),
        "patch_size": patch_size,
        "enc_mask_scale": (0.2, 0.8),
        "pred_mask_scale": (0.2, 0.8),
        "aspect_ratio": (0.3, 3.0),
        "nenc": 1,
        "npred": 2,
        "min_keep": 4,
        "allow_overlap": False,
        "data_transforms": [norm_img, img_channels_first],
    }

    #
    # Dataset
    #
    train_dataset_config = {
        "hdf5_file": "/home/turi/aulas/TCC/data/cifar-10-python/cifar10.hdf5",
        "group": "train",
        "data_dataset": "images",
        "labels_dataset": None,
    }
    val_dataset_config = {
        "hdf5_file": "/home/turi/aulas/TCC/data/cifar-10-python/cifar10.hdf5",
        "group": "val",
        "data_dataset": "images",
        "labels_dataset": None,
    }

    #
    # Trainer
    #
    trainer_config = {
        "encoder": encoder,
        "encoder_config": encoder_config,
        "predictor": predictor,
        "predictor_config": predictor_config,
        "batch_collator": batch_collator,
        "batch_collator_config": batch_collator_config,
        "model_name": "cifar_ijepa",
        "hdf5_dataset_train_config": train_dataset_config,
        "train_data_frac": 0.05,
        "hdf5_dataset_val_config": val_dataset_config,
        "val_data_frac": 0.05,
        "save_path": "./train_output",
        "params_to_save": "all",
        "seed": 42,
        "batch_size": 2,
        "epochs": 10,
        "start_lr": 1e-5,
        "ref_lr": 1e-4,
        "final_lr": 1e-6,
        "wd": 0.02,
        "final_wd": 0.0,
        "ipe_scale": 1.0,
        "warmup_epochs": 2,
        "opt_config": {},
        "master_addr": "localhost",
        "master_port": "4321",
        "backend": "nccl",
        "main_device": "cpu",
        "process_timeout": 100000,
    }
    trainer = DDPIJepaTrainer(**trainer_config)

    trainer.spawn_single_train()
