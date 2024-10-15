import torch
from report import ReportGenerator


class MockTrainer:
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size


# Mock models
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)


# Test case for the ReportGenerator
def test_report_generator():
    # Setup paths and mock data
    save_path = "./test_output"
    main_device = torch.device("cpu")
    metrics = {
        "accuracy": ["line1", "line3"],
        "loss": {"graph1": ["line1"]},
    }
    trainer = MockTrainer(lr=0.001, batch_size=32)
    params_to_save = ["lr", "batch_size"]
    best_metrics_obj = {"accuracy/line3": "max", "loss/graph1/line1": "min"}

    # Initialize ReportGenerator
    report_gen = ReportGenerator(
        save_path=save_path,
        main_device=main_device,
        metrics=metrics,
        trainer=trainer,
        params_to_save=params_to_save,
        best_metrics_obj=best_metrics_obj,
    )

    # Initialize mock models
    models = {"model1": MockModel(), "model2": MockModel()}

    # Simulate an epoch and update metrics
    report_gen.init_epoch_metrics_dict(epoch=0)
    report_gen.add_epoch_metric("accuracy/line1", 0.9)
    report_gen.add_epoch_metric("accuracy/line3", 0.7)
    # report_gen.add_epoch_metric("accuracy/graph2/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph3/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph4/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph5/line2", 0.85)
    report_gen.add_epoch_metric("loss/graph1/line1", 0.1)
    report_gen.add_epoch_metric("accuracy/line1", 0.7)
    report_gen.add_epoch_metric("accuracy/line3", 0.8)
    # report_gen.add_epoch_metric("accuracy/graph2/line2", 1.3)
    # report_gen.add_epoch_metric("accuracy/graph3/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph4/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph5/line2", 0.85)
    report_gen.add_epoch_metric("loss/graph1/line1", 0.1)

    report_gen.update_global_metrics()
    report_gen.save_models(models)
    report_gen.save_best_models(models)
    report_gen.save_metrics()
    report_gen.save_plots()

    report_gen.init_epoch_metrics_dict(epoch=1)
    report_gen.add_epoch_metric("accuracy/line1", 0.2)
    report_gen.add_epoch_metric("accuracy/line3", 0.3)
    # report_gen.add_epoch_metric("accuracy/graph2/line2", 0.9)
    # report_gen.add_epoch_metric("accuracy/graph3/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph4/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph5/line2", 0.85)
    report_gen.add_epoch_metric("loss/graph1/line1", 0.14)
    report_gen.add_epoch_metric("accuracy/line1", 0.8)
    report_gen.add_epoch_metric("accuracy/line3", 0.7)
    # report_gen.add_epoch_metric("accuracy/graph2/line2", 0.6)
    # report_gen.add_epoch_metric("accuracy/graph3/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph4/line2", 0.85)
    # report_gen.add_epoch_metric("accuracy/graph5/line2", 0.85)
    report_gen.add_epoch_metric("loss/graph1/line1", 0.2)

    report_gen.update_global_metrics()
    report_gen.save_models(models)
    report_gen.save_best_models(models)
    report_gen.save_metrics()
    report_gen.save_plots()

    # Save parameters, models, and metrics
    report_gen.save_params()


# Run the test
test_report_generator()
