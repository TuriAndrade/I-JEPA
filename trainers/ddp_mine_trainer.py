from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import HDF5Dataset
from report import ReportGenerator
from tqdm import tqdm
from optimizers import adamw_cosine_warmup_wd
from models import MINE
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import os
import pickle


class DDPMINETrainer:
    def __init__(
        self,
        model,
        model_ckpt_path,
        model_config_path,
        mine_config,
        batch_collator,
        batch_collator_config,
        model_name,
        load_pretrained_path,
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
        self.model_ckpt_path = model_ckpt_path
        self.model_config_path = model_config_path
        self.mine_config = mine_config
        self.batch_collator = batch_collator
        self.batch_collator_config = batch_collator_config
        self.model_name = model_name
        self.load_pretrained_path = load_pretrained_path
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

        with open(self.model_config_path, "rb") as f:
            self.model_config = pickle.load(f)

        self.init_model = None
        self.init_mine = None
        self.metrics = {
            "mi_lb": ["test"],
            "learning_rate": ["learning_rate"],
            "weight_decay": ["weight_decay"],
        }
        self.best_metrics_obj = {
            "mi_lb/test": "max",
        }

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

        self.init_model.load_state_dict(
            torch.load(
                self.model_ckpt_path,
                weights_only=True,
            ),
            strict=False,
        )

        self.init_mine = MINE(**self.mine_config)

    def launch_models(self, device, world_size):
        model = self.model(**self.model_config).to(device)
        model.load_state_dict(self.init_model.state_dict())

        mine = MINE(**self.mine_config).to(device)
        mine.load_state_dict(self.init_mine.state_dict())

        if world_size > 1:
            return DDP(mine, device_ids=[device]), model

        else:
            return mine, model

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
        self.report_generator.save_model_configs({self.model_name: self.mine_config})

        mp.spawn(
            self.train_ddp,
            args=(world_size,),
            nprocs=world_size,
        )

    def spawn_single_train(self):
        self.init_models()

        print("---- Initiating Single GPU training ----")
        self.report_generator = ReportGenerator(
            save_path=self.save_path,
            main_device=self.main_device,
            metrics=self.metrics,
            trainer=self,
            params_to_save=self.params_to_save,
            best_metrics_obj=self.best_metrics_obj,
        )
        self.report_generator.save_params(filename="ijepa_mine_params.json")
        self.report_generator.save_model_configs({self.model_name: self.mine_config})

        self.train(rank=self.main_device, world_size=1)

    def train(
        self,
        rank,
        world_size,
    ):
        test_loader = HDF5Dataset.get_dataloader(
            self.hdf5_dataset_test_config,
            batch_size=self.batch_size,
            num_workers=world_size,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            seed=self.seed,
            data_frac=self.test_data_frac,
            collate_fn=(
                self.batch_collator(**self.batch_collator_config)
                if self.batch_collator
                else None
            ),
        )

        mine, model = self.launch_models(rank, world_size)
        optimizer, _, scheduler, wd_scheduler = adamw_cosine_warmup_wd(
            models=mine,
            iterations_per_epoch=len(test_loader),
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

        for epoch in range(self.epochs):
            with tqdm(
                total=(len(test_loader)),
                desc=f"Epoch {epoch+1}",
                disable=(rank != self.main_device),
            ) as bar:
                self.report_generator.init_epoch_metrics_dict(
                    epoch=epoch,
                    device=rank,
                )

                mine.train()
                for data_batch, masks_ctx, masks_tgt in test_loader:
                    data_batch = data_batch.to(rank)
                    masks_ctx = [mask.to(rank) for mask in masks_ctx]
                    masks_tgt = [mask.to(rank) for mask in masks_tgt]

                    # 1. Zero grad
                    optimizer.zero_grad()

                    # 2. Fwd pass
                    with torch.no_grad():
                        ctx_out = model(data_batch, masks_ctx)
                        tgt_out = model(data_batch, masks_tgt)

                    # 2.1 Build joint and marginal batches
                    ctx_out = ctx_out.mean(dim=1)
                    tgt_out = tgt_out.mean(dim=1)

                    joint = torch.cat((ctx_out, tgt_out), dim=1)
                    marginal = torch.cat(
                        (ctx_out, tgt_out[torch.randperm(tgt_out.size(0))]), dim=1
                    )

                    # 2.2 Learn MINE
                    loss, mi_lb = mine(joint, marginal)

                    # 3. Backprop grad
                    loss.backward()

                    # 4. Update model
                    optimizer.step()

                    # 5. Update learning rate
                    lr = scheduler.step()

                    # 6. Update weight decay
                    wd = wd_scheduler.step()

                    self.report_generator.add_epoch_metric(
                        path="mi_lb/test",
                        value=mi_lb.item(),
                        device=rank,
                    )
                    bar.set_postfix(
                        {
                            "mi_lb": self.report_generator.get_last_epoch_metric(
                                path="mi_lb/test"
                            ),
                        }
                    )
                    bar.update(1)

                self.report_generator.update_global_metrics(device=rank)
                self.report_generator.save_metrics(
                    filename="ijepa_mine_metrics.json", device=rank
                )
                self.report_generator.save_plots(device=rank)
