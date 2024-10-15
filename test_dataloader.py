import torch
from dataloaders import HDF5Dataset
from tqdm import tqdm


def transform_data(data):
    return data / 255.0


# Configuration for the dataset
dataset_config = {
    "hdf5_file": "/home/ubuntu/JEPA_TCC/data/imagenet/imagenet_data.hdf5",  # Path to your HDF5 file
    "group": "train",  # Use the training group
    "data_dataset": "images",  # Dataset name for images
    "labels_dataset": "labels",  # Dataset name for labels
    "data_transform": transform_data,
}


def test_dataloader(rank, num_workers):
    print(
        f"\nTesting DataLoader on GPU {rank} with num_workers={num_workers} and world_size=4"
    )

    # Get the DataLoader
    loader = HDF5Dataset.get_dataloader(
        dataset_config,
        batch_size=128,  # You can adjust batch size as needed
        num_workers=num_workers,
        rank=rank,  # Rank for the current process
        world_size=1,  # Total number of GPUs
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        data_frac=0.01,
    )

    # Print the total number of batches
    total_batches = len(loader)
    print(f"Total number of batches in the loader for rank {rank}: {total_batches}")

    # Fetch a few batches and print their shapes
    for _ in tqdm(loader, total=len(loader), desc="Batches", disable=rank != 0):
        pass


if __name__ == "__main__":
    num_workers = 1  # Number of workers for data loading
    world_size = 1  # Total number of GPUs

    # Use torch.multiprocessing to spawn processes
    torch.multiprocessing.spawn(
        test_dataloader,
        args=(num_workers,),
        nprocs=world_size,
        join=True,
    )
