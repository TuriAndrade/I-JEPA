from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from dataloaders import HDF5Dataset
from report import CustomJSONEncoder
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
import pickle
import json


class ClassificationEvaluation:
    def __init__(
        self,
        model,
        model_ckpt_path,
        model_config_path,
        batch_collator,
        batch_collator_config,
        save_path,
        hdf5_dataset_test_config,
        test_data_frac,
        batch_size,
        seed,
        n_bootstraps,
        confidence_level,
    ):
        self.model = model
        self.model_ckpt_path = model_ckpt_path
        self.model_config_path = model_config_path
        self.batch_collator = batch_collator
        self.batch_collator_config = batch_collator_config
        self.save_path = save_path
        self.hdf5_dataset_test_config = hdf5_dataset_test_config
        self.batch_size = batch_size
        self.seed = seed
        self.test_data_frac = test_data_frac
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level

    def compute_metrics(self, y_true, y_pred_proba):
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0.0
        )
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
        auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")

        return accuracy, precision, recall, f1, auc

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

        # Load the model and its state
        with open(self.model_config_path, "rb") as f:
            model_config = pickle.load(f)
        model = self.model(**model_config).to(device)
        state_dict = torch.load(self.model_ckpt_path, weights_only=True)
        model.load_state_dict(state_dict)

        # Evaluate model on test
        test_labels = []
        test_preds = []

        with torch.no_grad():
            model.eval()
            with tqdm(
                total=len(test_loader), desc="Computing model output on test set"
            ) as bar:
                for data_batch, label_batch in test_loader:
                    data_batch = data_batch.to(device)
                    label_batch = label_batch.to(device)

                    model_out = model(data_batch)
                    test_probs = torch.softmax(model_out, dim=1)
                    test_preds.append(test_probs.cpu().numpy())
                    test_labels.append(label_batch.cpu().numpy())

                    bar.update(1)

        test_labels = np.concatenate(test_labels)
        test_preds = np.concatenate(test_preds)

        # Perform bootstrapping
        rng = np.random.RandomState(self.seed)

        bootstrapped_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

        for _ in tqdm(range(self.n_bootstraps), desc="Bootstrapping metrics"):
            # Sample with replacement
            indices = rng.choice(len(test_labels), len(test_labels), replace=True)
            y_true_boot = test_labels[indices]
            y_pred_proba_boot = test_preds[indices]

            # Compute metrics
            accuracy, precision, recall, f1, auc = self.compute_metrics(
                y_true_boot,
                y_pred_proba_boot,
            )
            # Append to lists
            bootstrapped_metrics["accuracy"].append(accuracy)
            bootstrapped_metrics["precision"].append(precision)
            bootstrapped_metrics["recall"].append(recall)
            bootstrapped_metrics["f1"].append(f1)
            bootstrapped_metrics["auc"].append(auc)

        # Now compute mean and confidence intervals for each metric
        alpha = 100 - self.confidence_level

        metrics_summary = {}
        for metric in bootstrapped_metrics:
            scores = np.array(bootstrapped_metrics[metric])
            mean = np.mean(scores)
            lower = np.percentile(scores, alpha / 2)
            upper = np.percentile(scores, 100 - alpha / 2)
            metrics_summary[metric] = {
                "mean": mean,
                "lower_ci": lower,
                "upper_ci": upper,
            }

        # Print the results
        print(f"Test Metrics with {self.confidence_level}% confidence intervals:")
        for metric in metrics_summary:
            mean = metrics_summary[metric]["mean"]
            lower = metrics_summary[metric]["lower_ci"]
            upper = metrics_summary[metric]["upper_ci"]
            print(
                f"{metric.capitalize()}: {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})"
            )

        # Save metrics to a CSV file
        metrics_dict = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            "Mean": [
                metrics_summary["accuracy"]["mean"],
                metrics_summary["precision"]["mean"],
                metrics_summary["recall"]["mean"],
                metrics_summary["f1"]["mean"],
                metrics_summary["auc"]["mean"],
            ],
            "Lower CI": [
                metrics_summary["accuracy"]["lower_ci"],
                metrics_summary["precision"]["lower_ci"],
                metrics_summary["recall"]["lower_ci"],
                metrics_summary["f1"]["lower_ci"],
                metrics_summary["auc"]["lower_ci"],
            ],
            "Upper CI": [
                metrics_summary["accuracy"]["upper_ci"],
                metrics_summary["precision"]["upper_ci"],
                metrics_summary["recall"]["upper_ci"],
                metrics_summary["f1"]["upper_ci"],
                metrics_summary["auc"]["upper_ci"],
            ],
        }

        metrics_df = pd.DataFrame(metrics_dict)
        metrics_csv_path = os.path.join(self.save_path, "test_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

        # Save model information to a json file
        model_info_path = os.path.join(self.save_path, "test_model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(
                {
                    "model": str(self.model),
                    "model_ckpt_path": self.model_ckpt_path,
                    "model_config_path": self.model_config_path,
                },
                f,
                cls=CustomJSONEncoder,
                indent=4,
            )

        print(f"Metrics saved to {metrics_csv_path}")
        print(f"Model info saved to {model_info_path}")
