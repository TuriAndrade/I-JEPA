from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import HDF5Dataset
from report import ReportGenerator
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from optimizers import adamw_cosine_warmup_wd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os


class DDPClassificationTrainer:
    def __init__(
        self,
        model,
        model_config,
        model_name,
        hdf5_dataset_train_config,
        train_data_frac,
        hdf5_dataset_val_config,
        val_data_frac,
        hdf5_dataset_test_config,
        test_data_frac,
        save_path,
        params_to_save,
        seed,
        batch_size,
        epochs,
        start_lr,
        ref_lr,
        final_lr,
        wd,
        final_wd,
        ipe_scale,
        warmup_epochs,
        opt_config,
        master_addr,
        master_port,
        backend,
        main_device,
        process_timeout,
    ):
        self.model = model
        self.model_config = model_config
        self.model_name = model_name
        self.hdf5_dataset_train_config = hdf5_dataset_train_config
        self.train_data_frac = train_data_frac
        self.hdf5_dataset_val_config = hdf5_dataset_val_config
        self.val_data_frac = val_data_frac
        self.hdf5_dataset_test_config = hdf5_dataset_test_config
        self.test_data_frac = test_data_frac
        self.save_path = save_path
        self.params_to_save = params_to_save
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.wd = wd
        self.final_wd = final_wd
        self.ipe_scale = ipe_scale
        self.warmup_epochs = warmup_epochs
        self.opt_config = opt_config
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.main_device = main_device
        self.process_timeout = process_timeout

        self.init_model = None
        self.metrics = {
            "loss": ["train", "val"],
            "learning_rate": ["learning_rate"],
            "weight_decay": ["weight_decay"],
        }
        self.best_metrics_obj = {
            "loss/val": "min",
        }

        # self.mp_manager = mp.Manager()
        # self.best_models = self.mp_manager.dict()

        os.makedirs(self.save_path, exist_ok=True)

    def ddp_setup(self, rank, world_size):
        init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            timeout=(
                None
                if not self.process_timeout
                else timedelta(seconds=self.process_timeout)
            ),
        )

    def ddp_cleanup(self):
        destroy_process_group()

    def init_models(self):
        self.init_model = self.model(**self.model_config)

    def launch_ddp_models(self, device):
        model = self.model(**self.model_config).to(device)
        model.load_state_dict(self.init_model.state_dict())

        return DDP(model, device_ids=[device])

    def train_ddp(self, rank, world_size):
        self.ddp_setup(rank, world_size)

        self.train(
            rank=rank,
            world_size=world_size,
        )

        self.ddp_cleanup()

    def spawn_train_ddp(self):
        self.init_models()

        world_size = torch.cuda.device_count()

        print("---- Initiating DDP training ----")
        print(f"CUDA device count: {world_size}")
        print(f"Master address: {self.master_addr}")
        print(f"Master port: {self.master_port}")

        self.report_generator = ReportGenerator(
            save_path=self.save_path,
            main_device=self.main_device,
            metrics=self.metrics,
            trainer=self,
            params_to_save=self.params_to_save,
            best_metrics_obj=self.best_metrics_obj,
        )
        self.report_generator.save_params()

        mp.spawn(
            self.train_ddp,
            args=(world_size,),
            nprocs=world_size,
        )

    def train(
        self,
        rank,
        world_size,
    ):
        train_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_train_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.train_data_frac,
        )

        val_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_val_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.val_data_frac,
        )

        model = self.launch_ddp_models(rank)
        optimizer, _, scheduler, wd_scheduler = adamw_cosine_warmup_wd(
            model=model,
            iterations_per_epoch=len(train_loader),
            start_lr=self.start_lr,
            ref_lr=self.ref_lr,
            warmup=self.warmup_epochs,
            num_epochs=self.epochs,
            final_lr=self.final_lr,
            wd=self.wd,
            final_wd=self.final_wd,
            ipe_scale=self.ipe_scale,
            opt_config=self.opt_config,
        )
        criterion = F.cross_entropy

        for epoch in range(self.epochs):
            with tqdm(
                total=(len(train_loader) + len(val_loader)),
                desc=f"Epoch {epoch+1}",
                disable=(rank != self.main_device),
            ) as bar:
                self.report_generator.init_epoch_metrics_dict(
                    epoch=epoch,
                    device=rank,
                )

                model.train()
                for data_batch, label_batch in train_loader:
                    data_batch = data_batch.to(rank)
                    label_batch = label_batch.to(rank)

                    # 1. Zero grad
                    optimizer.zero_grad()

                    # 2. Fwd pass
                    model_out = model(data_batch)

                    # 3. Compute loss
                    loss = criterion(model_out, label_batch)

                    # 4. Backprop grad
                    loss.backward()

                    # 5. Update model
                    optimizer.step()

                    # 6. Update learning rate
                    lr = scheduler.step()

                    # 7. Update weight decay
                    wd = wd_scheduler.step()

                    self.report_generator.add_epoch_metric(
                        path="loss/train",
                        value=loss.item(),
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="learning_rate",
                        value=lr,
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="weight_decay",
                        value=wd,
                        device=rank,
                    )
                    bar.set_postfix(
                        {
                            "train_loss": self.report_generator.get_last_epoch_metric(
                                path="loss/train"
                            ),
                            "val_loss": self.report_generator.get_last_epoch_metric(
                                path="loss/val"
                            ),
                        }
                    )
                    bar.update(1)

                with torch.no_grad():
                    model.eval()
                    for data_batch, label_batch in val_loader:
                        data_batch = data_batch.to(rank)
                        label_batch = label_batch.to(rank)

                        model_out = model(data_batch)
                        loss = criterion(model_out, label_batch)

                        self.report_generator.add_epoch_metric(
                            path="loss/val",
                            value=loss.item(),
                            device=rank,
                        )
                        bar.set_postfix(
                            {
                                "train_loss": self.report_generator.get_last_epoch_metric(
                                    path="loss/train"
                                ),
                                "val_loss": self.report_generator.get_last_epoch_metric(
                                    path="loss/val"
                                ),
                            }
                        )
                        bar.update(1)

                self.report_generator.update_global_metrics(device=rank)

                self.report_generator.save_models(
                    models={f"last_{self.model_name}": model.module},
                    device=rank,
                )
                self.report_generator.save_best_models(
                    models={self.model_name: model.module},
                    device=rank,
                )
                self.report_generator.save_metrics(device=rank)
                self.report_generator.save_plots(device=rank)

    def compute_metrics(self, y_true, y_pred_proba, threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        return accuracy, precision, recall, f1, auc

    def test(
        self,
        load_path,
        rank=0,
        world_size=1,
        metric="accuracy",
        n_thresholds=100,
    ):
        val_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_val_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.val_data_frac,
        )

        test_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_test_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.test_data_frac,
        )

        # Load the model and its state
        model = self.model(**self.model_config).to(rank)
        state_dict = torch.load(load_path, map_location=rank)
        model.load_state_dict(state_dict)

        # Evaluate model on validation set to find best threshold
        val_labels = []
        val_preds = []

        with torch.no_grad():
            model.eval()
            with tqdm(
                total=len(test_loader), desc="Computing model output on val set"
            ) as bar:
                for data_batch, label_batch in val_loader:
                    data_batch = data_batch.to(rank)
                    label_batch = label_batch.to(rank)

                    model_out = model(data_batch)
                    val_preds.append(model_out.cpu().numpy())
                    val_labels.append(label_batch.cpu().numpy())

                    bar.update(1)

        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)

        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        best_threshold = 0.5
        best_metric_value = 0

        with tqdm(
            total=len(thresholds), desc=f"Finding best threshold for {metric}"
        ) as bar:
            for threshold in thresholds:
                accuracy, precision, recall, f1, auc = self.compute_metrics(
                    val_labels, val_preds, threshold
                )

                if metric == "accuracy" and accuracy > best_metric_value:
                    best_metric_value = accuracy
                    best_threshold = threshold
                elif metric == "precision" and precision > best_metric_value:
                    best_metric_value = precision
                    best_threshold = threshold
                elif metric == "recall" and recall > best_metric_value:
                    best_metric_value = recall
                    best_threshold = threshold
                elif metric == "f1" and f1 > best_metric_value:
                    best_metric_value = f1
                    best_threshold = threshold
                elif metric == "auc" and auc > best_metric_value:
                    best_metric_value = auc
                    best_threshold = threshold

                bar.update(1)

        # Evaluate model on test set with best threshold
        test_labels = []
        test_preds = []

        with torch.no_grad():
            model.eval()
            with tqdm(
                total=len(test_loader), desc="Computing model output on test set"
            ) as bar:
                for data_batch, label_batch in test_loader:
                    data_batch = data_batch.to(rank)
                    label_batch = label_batch.to(rank)

                    model_out = model(data_batch)
                    test_preds.append(model_out.cpu().numpy())
                    test_labels.append(label_batch.cpu().numpy())

                    bar.update(1)

        test_labels = np.concatenate(test_labels)
        test_preds = np.concatenate(test_preds)

        accuracy, precision, recall, f1, auc = self.compute_metrics(
            test_labels, test_preds, best_threshold
        )

        print(f"Test Metrics @ Best Threshold ({best_threshold:.2f}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        test_output_dir = os.path.join(self.save_path, "test_output")
        os.makedirs(test_output_dir, exist_ok=True)

        # Save metrics to a CSV file
        metrics_dict = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            "Value": [accuracy, precision, recall, f1, auc],
            "Threshold": [best_threshold] * 5,
        }

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_csv_path = os.path.join(test_output_dir, "test_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Save load path information to a txt file
        load_path_txt = os.path.join(test_output_dir, "load_path.txt")
        with open(load_path_txt, "w") as f:
            f.write(f"Model loaded from: {load_path}\n")

        print(f"Metrics saved to {metrics_csv_path}")
        print(f"Model load path saved to {load_path_txt}")
