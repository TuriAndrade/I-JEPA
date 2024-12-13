from torch.distributed import init_process_group, destroy_process_group
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloaders import HDF5Dataset
from report import ReportGenerator
from tqdm import tqdm
from optimizers import adamw_cosine_warmup_wd
from models import apply_masks, repeat_interleave_batch, VICReg
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import os


class DDPIJepaTrainer:
    def __init__(
        self,
        encoder,
        encoder_config,
        predictor,
        predictor_config,
        batch_collator,
        batch_collator_config,
        vic_reg_config,
        model_name,
        hdf5_dataset_train_config,
        train_data_frac,
        hdf5_dataset_val_config,
        val_data_frac,
        save_path,
        params_to_save,
        local_best_window,
        seed,
        batch_size,
        epochs,
        start_lr,
        ref_lr,
        final_lr,
        wd,
        final_wd,
        ipe_scale,
        ema,
        warmup_epochs,
        save_ckpt_interval,
        save_predictor,
        opt_config,
        master_addr,
        master_port,
        backend,
        main_device,
        process_timeout,
    ):
        self.encoder = encoder
        self.encoder_config = encoder_config
        self.predictor = predictor
        self.predictor_config = predictor_config
        self.batch_collator = batch_collator
        self.batch_collator_config = batch_collator_config
        self.vic_reg_config = vic_reg_config
        self.model_name = model_name
        self.hdf5_dataset_train_config = hdf5_dataset_train_config
        self.train_data_frac = train_data_frac
        self.hdf5_dataset_val_config = hdf5_dataset_val_config
        self.val_data_frac = val_data_frac
        self.save_path = save_path
        self.params_to_save = params_to_save
        self.local_best_window = local_best_window
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.wd = wd
        self.final_wd = final_wd
        self.ipe_scale = ipe_scale
        self.ema = ema
        self.warmup_epochs = warmup_epochs
        self.save_ckpt_interval = save_ckpt_interval
        self.save_predictor = save_predictor
        self.opt_config = opt_config
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.main_device = main_device
        self.process_timeout = process_timeout

        self.metrics = {
            "loss": ["train", "val"],
            "vic_reg_losses": {
                "inv": ["train", "val"],
                "std": ["train", "val"],
                "cov": ["train", "val"],
            },
            "learning_rate": ["learning_rate"],
            "weight_decay": ["weight_decay"],
        }
        self.best_metrics_obj = {
            "loss/val": "min",
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
        self.init_encoder = self.encoder(**self.encoder_config)
        self.init_target = self.encoder(**self.encoder_config)
        self.init_predictor = self.predictor(**self.predictor_config)

        self.init_vic_reg = VICReg(
            self.init_encoder,
            self.init_target,
            self.init_predictor,
            **self.vic_reg_config,
        )

    def launch_models(self, device, world_size):
        vic_reg = VICReg(
            self.init_encoder,
            self.init_target,
            self.init_predictor,
            **self.vic_reg_config,
        ).to(device)
        vic_reg.load_state_dict(self.init_vic_reg.state_dict())

        if world_size > 1:
            vic_reg = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vic_reg)

            return DDP(vic_reg, device_ids=[device])

        else:
            return vic_reg

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
        self.report_generator.save_model_configs(
            {
                f"{self.model_name}_enc": self.encoder_config,
                f"{self.model_name}_pred": self.predictor_config,
            }
        )

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
        self.report_generator.save_params()
        self.report_generator.save_model_configs(
            {
                f"{self.model_name}_enc": self.encoder_config,
                f"{self.model_name}_pred": self.predictor_config,
            }
        )

        self.train(rank=self.main_device, world_size=1)

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
            collate_fn=(
                self.batch_collator(**self.batch_collator_config)
                if self.batch_collator
                else None
            ),
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
            collate_fn=(
                self.batch_collator(**self.batch_collator_config)
                if self.batch_collator
                else None
            ),
        )

        vic_reg = self.launch_models(rank, world_size)
        ipe = len(train_loader)
        optimizer, _, scheduler, wd_scheduler = adamw_cosine_warmup_wd(
            models=vic_reg,
            iterations_per_epoch=ipe,
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
        momentum_scheduler = (
            self.ema[0]
            + i * (self.ema[1] - self.ema[0]) / (ipe * self.epochs * self.ipe_scale)
            for i in range(int(ipe * self.epochs * self.ipe_scale) + 1)
        )

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

                vic_reg.train()

                for data_batch, masks_ctx, masks_tgt in train_loader:
                    data_batch = data_batch.to(rank)
                    masks_ctx = [mask.to(rank) for mask in masks_ctx]
                    masks_tgt = [mask.to(rank) for mask in masks_tgt]

                    # 1. Zero grad
                    optimizer.zero_grad()

                    # 2 Fwd pass
                    loss, inv_loss, std_loss, cov_loss = vic_reg(
                        data_batch, masks_ctx, masks_tgt
                    )

                    # 3. Compute gradients
                    loss.backward()

                    # 4. Update encoder weights
                    optimizer.step()

                    # 5. Update learning rate
                    lr = scheduler.step()

                    # 6. Update weight decay
                    wd = wd_scheduler.step()

                    # 7. Update target weights
                    m = next(momentum_scheduler)
                    (
                        vic_reg.module._update_target(m)
                        if world_size > 1
                        else vic_reg._update_target(m)
                    )

                    self.report_generator.add_epoch_metric(
                        path="loss/train",
                        value=loss.item(),
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="vic_reg_losses/inv/train",
                        value=inv_loss.item(),
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="vic_reg_losses/std/train",
                        value=std_loss.item(),
                        device=rank,
                    )
                    self.report_generator.add_epoch_metric(
                        path="vic_reg_losses/cov/train",
                        value=cov_loss.item(),
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
                    vic_reg.eval()

                    for data_batch, masks_ctx, masks_tgt in val_loader:
                        data_batch = data_batch.to(rank)
                        masks_ctx = [mask.to(rank) for mask in masks_ctx]
                        masks_tgt = [mask.to(rank) for mask in masks_tgt]

                        # 1. Fwd pass
                        loss, inv_loss, std_loss, cov_loss = vic_reg(
                            data_batch, masks_ctx, masks_tgt
                        )

                        self.report_generator.add_epoch_metric(
                            path="loss/val",
                            value=loss.item(),
                            device=rank,
                        )
                        self.report_generator.add_epoch_metric(
                            path="vic_reg_losses/inv/val",
                            value=inv_loss.item(),
                            device=rank,
                        )
                        self.report_generator.add_epoch_metric(
                            path="vic_reg_losses/std/val",
                            value=std_loss.item(),
                            device=rank,
                        )
                        self.report_generator.add_epoch_metric(
                            path="vic_reg_losses/cov/val",
                            value=cov_loss.item(),
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
                self.report_generator.save_metrics(device=rank)
                self.report_generator.save_plots(device=rank)

                if (epoch + 1) % self.save_ckpt_interval == 0:
                    self.report_generator.save_models(
                        models=(
                            {
                                f"{self.model_name}_enc_epoch_{epoch + 1}": (
                                    vic_reg.module.encoder
                                    if world_size > 1
                                    else vic_reg.encoder
                                ),
                                f"{self.model_name}_pred_epoch_{epoch + 1}": (
                                    vic_reg.module.predictor
                                    if world_size > 1
                                    else vic_reg.predictor
                                ),
                            }
                            if self.save_predictor
                            else {
                                f"{self.model_name}_enc_epoch_{epoch + 1}": (
                                    vic_reg.module.encoder
                                    if world_size > 1
                                    else vic_reg.encoder
                                )
                            }
                        ),
                        device=rank,
                    )
