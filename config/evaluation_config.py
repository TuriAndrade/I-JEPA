import os


def default_classification_evaluation(
    model,
    model_ckpt_path,
    model_config_path,
    batch_collator,
    batch_collator_config,
    hdf5_dataset_test_config,
    test_data_frac=1,
    save_path=os.path.join(os.environ.get("output_dir"), "clf_eval"),
    batch_size=128,
    seed=42,
    n_bootstraps=1000,
    confidence_level=95,
):
    return {
        "model": model,
        "model_ckpt_path": model_ckpt_path,
        "model_config_path": model_config_path,
        "batch_collator": batch_collator,
        "batch_collator_config": batch_collator_config,
        "hdf5_dataset_test_config": hdf5_dataset_test_config,
        "test_data_frac": test_data_frac,
        "save_path": save_path,
        "batch_size": batch_size,
        "seed": seed,
        "n_bootstraps": n_bootstraps,
        "confidence_level": confidence_level,
    }


def default_lidar_evaluation(
    model,
    model_ckpt_paths,
    model_config_path,
    model_ckpt_epochs,
    batch_collator,
    batch_collator_config,
    hdf5_dataset_test_config,
    lidar_config,
    test_data_frac=1,
    save_path=os.path.join(os.environ.get("output_dir"), "clf_eval"),
    batch_size=128,
    seed=42,
):
    return {
        "model": model,
        "model_ckpt_paths": model_ckpt_paths,
        "model_config_path": model_config_path,
        "model_ckpt_epochs": model_ckpt_epochs,
        "batch_collator": batch_collator,
        "batch_collator_config": batch_collator_config,
        "hdf5_dataset_test_config": hdf5_dataset_test_config,
        "lidar_config": lidar_config,
        "test_data_frac": test_data_frac,
        "save_path": save_path,
        "batch_size": batch_size,
        "seed": seed,
    }
