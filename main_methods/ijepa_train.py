from models import VisionTransformer, VisionTransformerPredictor
from trainers import DDPIJepaTrainer
from batch_collators import MBMaskCollator
from config import (
    batch_collators as batch_collators_config,
    dataloaders as dataloaders_config,
    models as models_config,
    trainers as trainers_config,
)
import os


def main(save_path=None, dataset_name="imagenet_100"):
    #
    # Dataset
    #
    dataset_config = dataloaders_config.get_dataset_config(dataset_name)
    path, img_size, patch_size = (
        dataset_config["path"],
        dataset_config["img_size"],
        dataset_config["default_patch_size"],
    )
    hdf5_dataset_train_config, hdf5_dataset_val_config, hdf5_dataset_test_config = (
        dataloaders_config.hdf5_dataset(
            path=path,
            with_labels=False,
        )
    )
    if save_path is None:
        save_path = os.path.join(os.environ.get("output_dir"), "ijepa", dataset_name)

    #
    # Encoder
    #
    encoder = VisionTransformer
    encoder_config = models_config.get_dataset_vit_config(
        dataset_name=dataset_name,
        img_size=img_size,
        patch_size=patch_size,
    )

    #
    # Predictor
    #
    predictor = VisionTransformerPredictor
    predictor_config = models_config.get_dataset_vit_predictor_config(
        dataset_name=dataset_name,
        img_size=img_size,
        patch_size=patch_size,
    )

    #
    # Masks
    #
    batch_collator = MBMaskCollator
    batch_collator_config = batch_collators_config.default_ijepa_multiblock_collator(
        img_size=img_size,
        patch_size=patch_size,
    )

    #
    # Trainer
    #
    trainer_config = trainers_config.default_ijepa_trainer(
        encoder=encoder,
        encoder_config=encoder_config,
        predictor=predictor,
        predictor_config=predictor_config,
        batch_collator=batch_collator,
        model_name="ijepa",
        batch_collator_config=batch_collator_config,
        hdf5_dataset_train_config=hdf5_dataset_train_config,
        train_data_frac=1,
        hdf5_dataset_val_config=hdf5_dataset_val_config,
        val_data_frac=1,
        batch_size=128,
        epochs=150,
        warmup_epochs=20,
        save_path=save_path,
        master_addr=os.environ.get("default_addr"),
        master_port=os.environ.get("default_port"),
    )
    trainer = DDPIJepaTrainer(**trainer_config)

    trainer.spawn_single_train()
