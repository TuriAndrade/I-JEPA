from models import (
    VisionTransformer,
    VisionTransformerCrossPredictor,
    VisionTransformerPredictor,
)
from trainers import DDPIJepaTrainer
from batch_collators import MBMaskCollator
from config import (
    batch_collator_config,
    dataloader_config,
    model_config,
    trainer_config,
)
import os


def main(
    save_path=None,
    dataset_name="imagenet_100",
    encoder_name="vit_tiny",
    predictor_name="vit_predictor_tiny",
    epochs=300,
    warmup_epochs=40,
    use_spectral_norm=False,
    cross_attn=False,
):
    #
    # Dataset
    #
    dataset_cfg = dataloader_config.get_dataset_config(dataset_name)
    path, img_size, patch_size = (
        dataset_cfg["path"],
        dataset_cfg["img_size"],
        dataset_cfg["default_patch_size"],
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
    # Encoder
    #
    encoder = VisionTransformer
    encoder_cfg = model_config.get_model_config(
        model_name=encoder_name,
        img_size=img_size,
        patch_size=patch_size,
        use_spectral_norm=bool(use_spectral_norm),
    )

    #
    # Predictor
    #
    predictor = (
        VisionTransformerCrossPredictor
        if bool(cross_attn)
        else VisionTransformerPredictor
    )
    predictor_cfg = model_config.get_model_config(
        model_name=predictor_name,
        img_size=img_size,
        patch_size=patch_size,
        use_spectral_norm=bool(use_spectral_norm),
    )

    #
    # Masks
    #
    batch_collator = MBMaskCollator
    batch_collator_cfg = batch_collator_config.get_collator_config(
        collator_name="default_ijepa_multiblock_collator",
        img_size=img_size,
        patch_size=patch_size,
    )

    #
    # Trainer
    #
    trainer_cfg = trainer_config.default_ijepa_trainer(
        encoder=encoder,
        encoder_config=encoder_cfg,
        predictor=predictor,
        predictor_config=predictor_cfg,
        batch_collator=batch_collator,
        model_name="ijepa",
        batch_collator_config=batch_collator_cfg,
        hdf5_dataset_train_config=hdf5_dataset_train_cfg,
        train_data_frac=1,
        hdf5_dataset_val_config=hdf5_dataset_val_cfg,
        val_data_frac=1,
        batch_size=128,
        save_ckpt_interval=10,
        save_predictor=False,
        epochs=int(epochs),
        warmup_epochs=int(warmup_epochs),
        save_path=save_path,
        master_addr=os.environ.get("default_addr"),
        master_port=os.environ.get("default_port"),
    )
    trainer = DDPIJepaTrainer(**trainer_cfg)

    trainer.spawn_train_ddp()
