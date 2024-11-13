from models import VisionTransformer
from trainers import DDPClassificationTrainer
from batch_collators import SupervisedCollator
from config import (
    batch_collators as batch_collators_config,
    dataloaders as dataloaders_config,
    models as models_config,
    trainers as trainers_config,
)
import os


def main(
    load_pretrained_path=None,
    save_path=None,
    dataset_name="imagenet_100",
    model_name="vit_tiny",
    data_frac=1,
    epochs=30,
    warmup_epochs=5,
):
    #
    # Dataset
    #
    dataset_config = dataloaders_config.get_dataset_config(dataset_name)
    path, img_size, patch_size, n_classes = (
        dataset_config["path"],
        dataset_config["img_size"],
        dataset_config["default_patch_size"],
        dataset_config["n_classes"],
    )
    hdf5_dataset_train_config, hdf5_dataset_val_config, hdf5_dataset_test_config = (
        dataloaders_config.hdf5_dataset(
            path=path,
            with_labels=True,
        )
    )
    if save_path is None:
        save_path = os.path.join(
            os.environ.get("output_dir"),
            (
                f"clf_{model_name}"
                if load_pretrained_path is None
                else f"pretrained_clf_{model_name}"
            ),
            dataset_name,
        )

    #
    # Model
    #
    model = VisionTransformer
    model_config = models_config.get_model_config(
        model_name=model_name,
        img_size=img_size,
        patch_size=patch_size,
        out_dim=n_classes,
    )

    #
    # Batch collator
    #
    batch_collator = SupervisedCollator
    batch_collator_config = batch_collators_config.get_collator_config(
        collator_name="default_classification_collator"
    )

    #
    # Trainer
    #
    trainer_config = trainers_config.default_classification_trainer(
        model=model,
        model_config=model_config,
        batch_collator=batch_collator,
        model_name="vit_clf",
        batch_collator_config=batch_collator_config,
        hdf5_dataset_train_config=hdf5_dataset_train_config,
        load_pretrained_path=load_pretrained_path,
        train_data_frac=data_frac,
        hdf5_dataset_val_config=hdf5_dataset_val_config,
        val_data_frac=data_frac,
        batch_size=128,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        save_path=save_path,
        master_addr=os.environ.get("default_addr"),
        master_port=os.environ.get("default_port"),
    )
    trainer = DDPClassificationTrainer(**trainer_config)

    trainer.spawn_train_ddp()
