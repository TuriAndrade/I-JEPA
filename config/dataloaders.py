import os


def imagenet1k_config():
    return {
        "path": os.environ.get("imagenet1k_path"),
        "img_size": 224,
        "default_patch_size": 16,
    }


def imagenet100_config():
    return {
        "path": os.environ.get("imagenet100_path"),
        "img_size": 224,
        "default_patch_size": 16,
    }


def cifar10_config():
    return {
        "path": os.environ.get("cifar10_path"),
        "img_size": 32,
        "default_patch_size": 2,
    }


def hdf5_dataset(
    path=os.environ.get("imagenet1k_path"),
    with_labels=True,
):
    train_dataset_config = {
        "hdf5_file": path,
        "group": "train",
        "data_dataset": "images",
        "labels_dataset": "labels" if with_labels else None,
    }
    val_dataset_config = {
        "hdf5_file": path,
        "group": "val",
        "data_dataset": "images",
        "labels_dataset": "labels" if with_labels else None,
    }
    test_dataset_config = {
        "hdf5_file": path,
        "group": "test",
        "data_dataset": "images",
        "labels_dataset": "labels" if with_labels else None,
    }

    return train_dataset_config, val_dataset_config, test_dataset_config
