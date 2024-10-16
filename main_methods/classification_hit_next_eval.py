from evaluation import ClassificationEvaluation
from models import HiTNeXt
from dataloaders import norm_img


def main(
    model_ckpt_path,
    model_config_path,
    train_output_dir,
):

    #
    # Model
    #
    model = HiTNeXt

    #
    # Dataloader
    #
    test_dataset_config = {
        "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",
        "group": "test",
        "data_dataset": "images",
        "labels_dataset": "labels",
        "data_transform": norm_img,
    }

    #
    # Evaluation
    #
    evaluator = ClassificationEvaluation(
        model=model,
        model_ckpt_path=model_ckpt_path,
        model_config_path=model_config_path,
        save_path=train_output_dir,
        hdf5_dataset_test_config=test_dataset_config,
        test_data_frac=1.0,
        batch_size=32,
        seed=42,
    )

    #
    # Evaluation
    #
    evaluator.test()
