from models import HiTNeXt, HiTNextConfig, LayerNorm
from trainers.classification import DDPClassificationTrainer
from dataloaders import HDF5Dataset
import torch


def main():
    #
    # Model
    #
    model = HiTNeXt
    model_config = HiTNextConfig(
        img_size=256,
        in_chans=3,
        channels_last=True,
        n_stages=4,
        embed_dim=(96, 192, 384, 768),
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        ape=False,
        rpe_type="rpe_default",
        use_checkpoint=False,
        pretrained_window_size=8,
        apply_out_head=False,
        out_head_dim=None,
        patch_embed_config={
            "conv_config": {
                "kernel_size": 4,
                "stride": 4,
            },
            "conv_dw": True,
            "drop_path": 0.0,
            "use_act_block": True,
            "use_ln": True,
            "ln_before": False,
            "resid": True,
        },
        patch_merge_config={
            "conv_config": {
                "kernel_size": 2,
                "stride": 2,
            },
            "conv_dw": True,
            "drop_path": 0.0,
            "use_act_block": True,
            "use_ln": True,
            "ln_before": False,
            "resid": True,
        },
    )

    #
    # Dataloader
    #
    def transform_data(data):
        return data / 255.0

    train_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "train",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": transform_data,
    }
    val_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "val",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": transform_data,
    }
    test_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "test",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": transform_data,
    }

    #
    # Trainer
    #
    trainer = DDPClassificationTrainer(
        model=model,
        model_config=model_config.__dict__,
        model_name="hit_next",
        hdf5_dataset_train_config=train_dataset_config,
        hdf5_dataset_val_config=val_dataset_config,
        hdf5_dataset_test_config=test_dataset_config,
        save_path="/home/ubuntu/JEPA_TCC/train_output/",
        params_to_save=[""],
        batch_size=128,
        epochs=50,
        start_lr=5e-7,
        ref_lr=5e-4,
        final_lr=5e-6,
        wd=0.05,
        final_wd=0.5,
        warmup_epochs=5,
        master_addr="localhost",
        master_port="4321",
        backend="nccl",
        main_device=0,
        process_timeout=10000,
    )
