from models import VisionTransformer
from trainers import DDPClassificationTrainer
from batch_collators import SupervisedCollator
from config import (
    batch_collator_config,
    dataloader_config,
    model_config,
    trainer_config,
)
import os


def main(
    load_pretrained_path=None,
    save_path=None,
    dataset_name="imagenet_100",
    model_name="vit_tiny",
    data_frac=1,
    epochs=20,
    warmup_epochs=3,
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
    model_cfg = model_config.get_model_config(
        model_name=model_name,
        img_size=img_size,
        patch_size=patch_size,
        out_dim=n_classes,
    )

    #
    # Batch collator
    #
    batch_collator = SupervisedCollator
    batch_collator_cfg = batch_collator_config.get_collator_config(
        collator_name="default_classification_collator"
    )

    #
    # Trainer
    #
    trainer_cfg = trainer_config.default_classification_trainer(
        model=model,
        model_config=model_cfg,
        batch_collator=batch_collator,
        model_name="vit_clf",
        batch_collator_config=batch_collator_cfg,
        hdf5_dataset_train_config=hdf5_dataset_train_cfg,
        load_pretrained_path=load_pretrained_path,
        train_data_frac=data_frac,
        hdf5_dataset_val_config=hdf5_dataset_val_cfg,
        val_data_frac=data_frac,
        batch_size=128,
        epochs=int(epochs),
        warmup_epochs=int(warmup_epochs),
        save_path=save_path,
        master_addr=os.environ.get("default_addr"),
        master_port=os.environ.get("default_port"),
    )
    trainer = DDPClassificationTrainer(**trainer_cfg)

    trainer.spawn_train_ddp()
