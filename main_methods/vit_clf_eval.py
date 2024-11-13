from models import VisionTransformer
from evaluation import ClassificationEvaluation
from batch_collators import SupervisedCollator
from config import (
    batch_collators as batch_collators_config,
    dataloaders as dataloaders_config,
    evaluation as evaluation_config,
)
import os


def main(
    model_ckpt_path,
    model_config_path,
    save_path=None,
    dataset_name="imagenet_100",
    model_name="vit_tiny",
    pretrained=False,
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
            f"clf_{model_name}" if not pretrained else f"pretrained_clf_{model_name}",
            dataset_name,
        )

    #
    # Model
    #
    model = VisionTransformer

    #
    # Batch collator
    #
    batch_collator = SupervisedCollator
    batch_collator_config = batch_collators_config.get_collator_config(
        collator_name="default_classification_collator"
    )

    #
    # Eval
    #
    evaluator_config = evaluation_config.default_classification_evaluation(
        model=model,
        model_ckpt_path=model_ckpt_path,
        model_config_path=model_config_path,
        batch_collator=batch_collator,
        batch_collator_config=batch_collator_config,
        save_path=save_path,
        hdf5_dataset_test_config=hdf5_dataset_test_config,
        batch_size=128,
        seed=42,
        n_bootstraps=1000,
        confidence_level=95,
    )
    evaluator = ClassificationEvaluation(**evaluator_config)
    evaluator.test()
