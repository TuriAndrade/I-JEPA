from models import VisionTransformer
from evaluation import LiDAREvaluation
from batch_collators import MBMaskCollator
from config import (
    batch_collator_config,
    dataloader_config,
    evaluation_config,
)
import os


def main(
    model_ckpt_dir,
    model_config_path,
    batch_size=128,
    save_path=None,
    dataset_name="imagenet_100",
    encoder_name="vit_tiny",
    epochs=300,
    epochs_interval=10,
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
            os.environ.get("output_dir"), f"ijepa_{encoder_name}", dataset_name
        )

    #
    # Model
    #
    model = VisionTransformer

    #
    # Batch collator
    #
    batch_collator = MBMaskCollator
    batch_collator_cfg = batch_collator_config.get_collator_config(
        collator_name="default_ijepa_multiblock_collator",
        img_size=img_size,
        patch_size=patch_size,
        n_enc=20,
        n_pred=1,
    )

    #
    # Eval
    #
    evaluator_cfg = evaluation_config.default_lidar_evaluation(
        model=model,
        model_ckpt_paths=[
            os.path.join(model_ckpt_dir, f"ijepa_enc_epoch_{epoch}.pt")
            for epoch in range(epochs_interval, epochs + 1, epochs_interval)
        ],
        model_config_path=model_config_path,
        model_ckpt_epochs=list(range(epochs_interval, epochs + 1, epochs_interval)),
        batch_collator=batch_collator,
        batch_collator_config=batch_collator_cfg,
        save_path=save_path,
        hdf5_dataset_test_config=hdf5_dataset_test_cfg,
        test_data_frac=1,
        lidar_config={},
        batch_size=batch_size,
        seed=42,
    )
    evaluator = LiDAREvaluation(**evaluator_cfg)
    evaluator.test()
