from dataloaders import HDF5Dataset
from .utils import LiDAR
from report import CustomJSONEncoder
from tqdm import tqdm
import torch
import os
import pickle
import json
import numpy as np


class LiDAREvaluation:
    def __init__(
        self,
        model,
        model_ckpt_paths,
        model_config_path,
        model_ckpt_epochs,
        batch_collator,
        batch_collator_config,
        save_path,
        hdf5_dataset_test_config,
        lidar_config,
        test_data_frac,
        batch_size,
        seed,
    ):
        self.model = model
        self.model_ckpt_paths = model_ckpt_paths
        self.model_config_path = model_config_path
        self.model_ckpt_epochs = model_ckpt_epochs
        self.batch_collator = batch_collator
        self.batch_collator_config = batch_collator_config
        self.save_path = save_path
        self.hdf5_dataset_test_config = hdf5_dataset_test_config
        self.lidar_config = lidar_config
        self.batch_size = batch_size
        self.seed = seed
        self.test_data_frac = test_data_frac

    def test(
        self,
        device=0,
    ):
        test_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_test_config,
            batch_size=self.batch_size,
            num_workers=1,
            world_size=1,
            rank=device,
            shuffle=True,
            seed=self.seed,
            data_frac=self.test_data_frac,
            collate_fn=(
                self.batch_collator(**self.batch_collator_config)
                if self.batch_collator
                else None
            ),
        )

        lidar = LiDAR(**self.lidar_config)
        lidar_values = []
        lidar_upper_bound = 0
        with tqdm(total=len(self.model_ckpt_paths), desc="Computing LiDAR") as bar:
            for model_ckpt_path in self.model_ckpt_paths:
                # Load the model and its state
                with open(self.model_config_path, "rb") as f:
                    model_config = pickle.load(f)
                model = self.model(**model_config).to(device)
                state_dict = torch.load(model_ckpt_path, weights_only=True)
                model.load_state_dict(state_dict)

                # Compute LiDAR on test
                embeddings = []

                with torch.no_grad():
                    model.eval()
                    for data_batch, masks_ctx, _ in test_loader:
                        data_batch = data_batch.to(device)
                        masks_ctx = [mask.to(device) for mask in masks_ctx]

                        # 1. Get model output. (B*n_masks, N, D)
                        model_out = model(data_batch, masks_ctx)

                        # 2. Compute mean over seq len dim. (B*n_masks, D)
                        mean_embds = model_out.mean(dim=1)

                        # 3. Split batch into different masks. (n_masks, B, D)
                        mean_embds = mean_embds.split(self.batch_size, dim=0)

                        # 4. Stack different views over dim 1. (B, n_masks, D)
                        mean_embds = torch.stack(mean_embds, dim=1)

                        # 5. Append to 'embeddings'
                        embeddings.append(mean_embds)

                    embeddings = torch.cat(embeddings, dim=0)
                    lidar_value, _, lidar_upper_bound = lidar.run(
                        embeddings.cpu().numpy()
                    )
                    lidar_values.append(lidar_value)

                    bar.update(1)

        # Plot lidar
        LiDAR.plot(
            epochs=self.model_ckpt_epochs,
            lidar_values=lidar_values,
            upper_bound=lidar_upper_bound,
            output_file=os.path.join(self.save_path, "test_lidar_eval_plot.png"),
        )

        # Save model information to a json file
        best_model_idx = np.argmax(lidar_values)
        model_info_path = os.path.join(self.save_path, "test_lidar_eval_info.json")
        with open(model_info_path, "w") as f:
            json.dump(
                {
                    "model": str(self.model),
                    "best_lidar_ckpt_path": self.model_ckpt_paths[best_model_idx],
                    "best_lidar_config_path": self.model_config_path,
                },
                f,
                cls=CustomJSONEncoder,
                indent=4,
            )
