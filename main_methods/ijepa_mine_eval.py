from models import VisionTransformer
from trainers import DDPMINETrainer
from batch_collators import MBMaskCollator
from config import (
    batch_collator_config,
    dataloader_config,
    model_config,
    trainer_config,
)
import os


def main(
    model_ckpt_path,
    model_config_path,
    save_path=None,
    dataset_name="imagenet_100",
    model_name="vit_tiny",
    mine_name="mine_vit_tiny",
    data_frac=1,
    epochs=1000,
    batch_size=32,
    warmup_epochs=3,
    ddp=True,
):
    #
    # Dataset
    #
    dataset_cfg = dataloader_config.get_dataset_config(dataset_name)
    path, img_size, patch_size, n_classes = (
        dataset_cfg["path"],
        dataset_cfg["img_size"],
        dataset_cfg["default_patch_size"],
        dataset_cfg["n_classes"],
    )
    hdf5_dataset_train_cfg, hdf5_dataset_val_cfg, hdf5_dataset_test_cfg = (
        dataloader_config.hdf5_dataset(
            path=path,
            with_labels=False,
        )
    )
    if save_path is None:
        save_path = os.path.join(
            os.environ.get("output_dir"),
            f"ijepa_mine_{model_name}",
            dataset_name,
        )

    #
    # Encoder
    #
    model = VisionTransformer

    #
    # MINE
    #
    mine_cfg = model_config.get_model_config(
        model_name=mine_name,
    )

    #
    # Masks
    #
    batch_collator = MBMaskCollator
    batch_collator_cfg = batch_collator_config.get_collator_config(
        collator_name="default_ijepa_multiblock_collator",
        img_size=img_size,
        patch_size=patch_size,
        n_enc=1,
        n_pred=1,
    )

    #
    # Trainer
    #
    trainer_cfg = trainer_config.default_mine_trainer(
        model=model,
        model_ckpt_path=model_ckpt_path,
        model_config_path=model_config_path,
        mine_config=mine_cfg,
        batch_collator=batch_collator,
        model_name="ijepa_mine_eval",
        batch_collator_config=batch_collator_cfg,
        hdf5_dataset_test_config=hdf5_dataset_test_cfg,
        test_data_frac=data_frac,
        batch_size=batch_size,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        save_path=save_path,
        master_addr=os.environ.get("default_addr"),
        master_port=os.environ.get("default_port"),
    )
    trainer = DDPMINETrainer(**trainer_cfg)

    if ddp:
        trainer.spawn_train_ddp()

    else:
        trainer.spawn_single_train()
