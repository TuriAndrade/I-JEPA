import torch
from dataloaders import HDF5Dataset

# Configuration for the dataset
dataset_config = {
    "hdf5_file": "/home/turi/aulas/TCC/data/cifar-10-python/cifar10.hdf5",  # Path to your HDF5 file
    "group": "train",  # Use the training group
    "data_dataset": "images",  # Dataset name for images
    "labels_dataset": "labels",  # Dataset name for labels
    "data_transform": lambda data: torch.permute(data, (2, 0, 1)) / 255.0,
}


def test_dataloader(num_workers):
    print(f"\nTesting DataLoader with num_workers={num_workers} and world_size=1")

    # Get the DataLoader
    loader = HDF5Dataset.get_dataloader(
        dataset_config,
        batch_size=128,  # You can adjust batch size as needed
        num_workers=num_workers,
        rank=0,  # Rank is 0 for single GPU
        world_size=1,  # World size is 1 for testing
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # Print the total number of batches
    total_batches = len(loader)
    print(f"Total number of batches in the loader: {total_batches}")

    # Fetch a few batches and print their shapes
    for i, (data, labels) in enumerate(loader):
        print(f"Batch {i + 1}: data shape: {data.shape}, labels shape: {labels.shape}")
        if i == 10:  # Just print first three batches
            break


# Test with 1 worker
test_dataloader(num_workers=1)

# Test with 2 workers
test_dataloader(num_workers=2)
