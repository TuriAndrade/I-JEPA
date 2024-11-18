from models import VisionTransformer
from evaluation import ClassificationEvaluation
from batch_collators import SupervisedCollator
from config import (
    batch_collator_config,
    dataloader_config,
    evaluation_config,
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
    batch_collator_cfg = batch_collator_config.get_collator_config(
        collator_name="default_classification_collator"
    )

    #
    # Eval
    #
    evaluator_cfg = evaluation_config.default_classification_evaluation(
        model=model,
        model_ckpt_path=model_ckpt_path,
        model_config_path=model_config_path,
        batch_collator=batch_collator,
        batch_collator_config=batch_collator_cfg,
        save_path=save_path,
        hdf5_dataset_test_config=hdf5_dataset_test_cfg,
        batch_size=128,
        seed=42,
        n_bootstraps=1000,
        confidence_level=95,
    )
    evaluator = ClassificationEvaluation(**evaluator_cfg)
    evaluator.test()
