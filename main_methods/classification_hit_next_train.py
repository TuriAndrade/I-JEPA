from models import HiTNeXt, HiTNextConfig, LayerNorm
from trainers import DDPClassificationTrainer
from dataloaders import norm_img
from datetime import datetime


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
        apply_out_head=True,
        out_head_dim=1000,
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
    train_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "train",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": norm_img,
    }
    val_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "val",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": norm_img,
    }

    #
    # Trainer
    #
    current_time = datetime.now()
    formatted_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")
    save_path = f"/home/ubuntu/JEPA_TCC/train_output/{formatted_time}/"
    trainer = DDPClassificationTrainer(
        model=model,
        model_config=model_config.__dict__,
        model_name="hit_next",
        hdf5_dataset_train_config=train_dataset_config,
        train_data_frac=0.1,
        hdf5_dataset_val_config=val_dataset_config,
        val_data_frac=1.0,
        save_path=save_path,
        params_to_save=[
            "model_config",
            "model_name",
            "hdf5_dataset_train_config",
            "hdf5_dataset_val_config",
            "batch_size",
            "epochs",
            "start_lr",
            "ref_lr",
            "final_lr",
            "wd",
            "final_wd",
            "warmup_epochs",
            "ipe_scale",
            "opt_config",
            "master_addr",
            "master_port",
            "backend",
            "main_device",
        ],
        batch_size=32,
        epochs=100,
        seed=0,
        start_lr=5e-7,
        ref_lr=5e-4,
        final_lr=5e-6,
        wd=0.05,
        final_wd=0.5,
        warmup_epochs=20,
        ipe_scale=1.25,
        opt_config={},
        master_addr="localhost",
        master_port="4321",
        backend="nccl",
        main_device=0,
        process_timeout=None,
    )

    #
    # Train
    #
    trainer.spawn_train_ddp()
