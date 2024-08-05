import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.dynamics.RegressionLossEvaluator import RegressionLossEvaluator
from loss.motion.RegressionLossEvaluator import RegressionLossEvaluator as MotionLoss
from cli.utilities import get_git_hash, has_uncommitted_changes
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging 
import functools
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--no-wandb', action='store_true', default=False,
                               help='Log this run to Weights and Biases.')
        subparser.add_argument('--model-type', type=str, default='feedforward', choices=['analytical', 'feedforward', 'groundlink', 'mdm'], help='The model to train.')
        subparser.add_argument('--output-data-format', type=str, default='all_frames', choices=['all_frames', 'last_frame'], help='Output for all frames in a window or only the last frame.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                               help='The path to a model checkpoint to save during training. Also, starts from the '
                                    'latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, 
                               help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=1,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--learning-rate', type=float, default=1e-4,
                               help='The learning rate for weight updates.')
        subparser.add_argument('--dropout', action='store_true', help='Apply dropout?')
        subparser.add_argument('--dropout-prob', type=float, default=0.5, help='Dropout prob')
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='sigmoid', help='Which activation func?')
        subparser.add_argument('--epochs', type=int, default=10, help='The number of epochs to run training for.')
        subparser.add_argument('--opt-type', type=str, default='rmsprop',
                               help='The optimizer to use when adapting the weights of the model during training.')
        subparser.add_argument('--batch-size', type=int, default=32,
                               help='The batch size to use when training the model.')
        subparser.add_argument('--short', action='store_true',
                               help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--data-loading-workers', type=int, default=3,
                               help='Number of separate processes to spawn to load data in parallel.')
        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)],
                               help='Which wrench components to train.')
        subparser.add_argument('--trial-filter', type=str, nargs='+', default=[""],
                               help='What kind of trials to train/test on.')
        subparser.add_argument('--use-diffusion', action='store_true', help='Use diffusion?')
        subparser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
        subparser.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
        subparser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
        subparser.add_argument('--schedule-sampler', default='uniform',
                               choices=['uniform','loss-second-moment'], help='Diffusion timestep sampler')
        subparser.add_argument("--lambda_pos", default=1.0, type=float, help="Joint positions loss.")
        subparser.add_argument("--lambda_vel", default=1.0, type=float, help="Joint velocity loss.")
        subparser.add_argument("--lambda_acc", default=1.0, type=float, help="Joint acceleration loss")
        subparser.add_argument("--lambda_fc", default=1.0, type=float, help="Foot contact loss.")
        subparser.add_argument("--lambda_wrench", default=1.0, type=float, help="Ground contact wrenches in root frame loss.")
        subparser.add_argument("--lambda_res_wrench", default=1.0, type=float, help="Residual wrenches in root frame loss.")


    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        output_data_format: str = args.output_data_format #'all_frames' if args.model_type == 'groundlink' else 'last_frame'
        opt_type: str = args.opt_type
        checkpoint_dir: str = os.path.join(model_type, os.path.abspath(args.checkpoint_dir))
        history_len: int = args.history_len
        root_history_len: int = 10
        hidden_dims: List[int] = args.hidden_dims
        learning_rate: float = args.learning_rate
        epochs: int = args.epochs
        batch_size: int = args.batch_size    
        device: str = args.device                                               
        short: bool = args.short
        dataset_home: str = args.dataset_home
        log_to_wandb: bool = not args.no_wandb
        data_loading_workers: int = args.data_loading_workers
        stride: int = args.stride
        batchnorm: bool = args.batchnorm
        dropout: bool = args.dropout
        dropout_prob: float = args.dropout_prob
        activation: str = args.activation
        geometry = self.ensure_geometry(args.geometry_folder)

        use_diffusion = args.use_diffusion
        self.noise_schedule = args.noise_schedule
        self.diffusion_steps = args.diffusion_steps
        self.sigma_small = args.sigma_small
        self.lambda_pos = args.lambda_pos
        self.lambda_vel = args.lambda_vel
        self.lambda_fc = args.lambda_fc
        self.lambda_acc = args.lambda_acc
        self.lambda_wrench = args.lambda_wrench
        self.lambda_res_wrench = args.lambda_res_wrench
        self.batch_size = batch_size


        # Initialize multiprocessing
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=5))
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)


        if log_to_wandb:
            # Grab all cmd args and add current git hash
            config = args.__dict__
            config["git_hash"] = get_git_hash

            logging.info('Initializing wandb...')
            # Check if WANDB_RUN_GROUP environment variable exists
            wandb_group = os.getenv('WANDB_RUN_GROUP', f'ddp_{wandb.util.generate_id()}')  # Default to 'DDP' if not set
            wandb.init(
                # set the wandb project where this run will be logged
                project="dynamics-diffusion",

                # track hyperparameters and run metadata
                config=config,
                group=wandb_group
            )


        # Create an instance of the dataset
        DEV = 'test'
        train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(dataset_home, DEV))

        print(f'Running on {torch.cuda.device_count()} gpus')
        print('Initializing datasets...')
        train_dataset = AddBiomechanicsDataset(train_dataset_path, history_len, device=torch.device(device), stride=stride, output_data_format=output_data_format,
                                               geometry_folder=geometry, testing_with_short_dataset=short)
        train_sampler = DS(train_dataset, drop_last=True, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True, sampler=train_sampler, drop_last=True)

        dev_dataset = AddBiomechanicsDataset(dev_dataset_path, history_len, device=torch.device(device), stride=stride, output_data_format=output_data_format,
                                             geometry_folder=geometry, testing_with_short_dataset=short)
        dev_sampler = DS(dev_dataset, shuffle=False, drop_last=True, rank=rank)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True, sampler=dev_sampler, drop_last=True)
        
        # Choose the correct evaluator
        LossEvaluator = MotionLoss if model_type == 'mdm' else RegressionLossEvaluator
        train_loss_evaluator = LossEvaluator(dataset=train_dataset, split='train', device=device)
        dev_loss_evaluator = LossEvaluator(dataset=dev_dataset, split=DEV, device=device)


        # Create an instance of the model
        print('Initializing model...')
        model = self.get_model(train_dataset.num_dofs,
                               train_dataset.num_contact_bodies,
                               model_type,
                               history_len=history_len,
                               stride=stride,
                               hidden_dims=hidden_dims,
                               activation=activation,
                               batchnorm=batchnorm,
                               dropout=dropout,
                               dropout_prob=dropout_prob,
                               root_history_len=root_history_len,
                               output_data_format=output_data_format,
                               device=rank).to(device)
        
        # Wrap model in DDP class
        ddp_model = DDP(model, device_ids=[device], find_unused_parameters=True)

        if use_diffusion:
            print('Initializing diffusion...')
            self.diffusion = self.get_gaussian_diffusion(args)
            self.schedule_sampler  = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)
            self.device = device
            self.ddp_model = ddp_model
            self.rank = rank
            self.log_to_wandb = log_to_wandb
            self.train_data_len, self.dev_data_len = len(train_dataloader), len(dev_dataloader)

        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        if not list(params_to_optimize):
            print("No parameters to optimize. Skipping training loop.")
            return False
        
        # Define the optimizer
        if opt_type == 'adagrad':
            optimizer = torch.optim.Adagrad(ddp_model.parameters(), lr=learning_rate)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)
        elif opt_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(ddp_model.parameters(), lr=learning_rate)
        elif opt_type == 'adadelta':
            optimizer = torch.optim.Adadelta(ddp_model.parameters(), lr=learning_rate)
        elif opt_type == 'adamax':
            optimizer = torch.optim.Adamax(ddp_model.parameters(), lr=learning_rate)
        else:
            logging.error('Invalid optimizer type: ' + opt_type)
            assert (False)

        self.load_latest_checkpoint(ddp_model, checkpoint_dir=checkpoint_dir, optimizer=optimizer)


        for epoch in range(epochs):
            print('-' * 80)
            print(f'\nEvaluating Dev Set before epoch {epoch}')
            print('-' * 80)
            train_sampler.set_epoch(epoch)
            """
            Diffusion Loop
            """
            if use_diffusion:
                with torch.no_grad():
                    ddp_model.eval()
                    for i, batch in enumerate(dev_dataloader):
                        inputs: Dict[str, torch.Tensor]
                        labels: Dict[str, torch.Tensor]
                        batch_subject_indices: List[int]
                        batch_trial_indices: List[int]
                        inputs, labels, batch_subject_indices, batch_trial_indices = batch

                        # Run the forward and backward process
                        self.diffusion_process(inputs, 'dev', i)
                        if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                            logging.info(f'  - [{rank=}] Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
            
                dist.barrier()
                ddp_model.train()
                if rank == 0: 
                    print('-' * 80)
                    print(f'Running Train Epoch {epoch}')    
                    print('-' * 80)   
                for i, batch in enumerate(train_dataloader):
                    optimizer.zero_grad()

                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    batch_trial_indices: List[int]
                    inputs, labels, batch_subject_indices, batch_trial_indices = batch

                    # Run the forward and backward process
                    self.diffusion_process(inputs, 'train', i)
                    if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                        logging.info(f'  - [{rank=}] Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))

                    optimizer.step()

                    if (i + 1) % 1000 == 0 or i == len(train_dataloader) - 1 and rank == 0:
                        model_path = f"{checkpoint_dir}/epoch_{epoch}_batch_{i}.pt"
                        os.makedirs(os.path.dirname(model_path), exist_ok=True) 
                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': ddp_model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                    }, model_path)
                continue
                

            """
            Normal Loop
            """
            with torch.no_grad():
                ddp_model.eval()  # Turn dropout off
                for i, batch in enumerate(dev_dataloader):
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    batch_trial_indices: List[int]
                    inputs, labels, batch_subject_indices, batch_trial_indices = batch

                    outputs = ddp_model(inputs)

                    # Ensure logging is only done on rank 0 process and calculation
                    # is synchronized
                    dev_loss_evaluator(inputs,
                                        outputs,
                                        labels,
                                        batch_subject_indices,
                                        batch_trial_indices,
                                        args,
                                        compute_report=True)
                    if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                        print('  - Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
                # Report dev loss on this epoch
                if rank == 0: 
                    print('Dev Set Evaluation:')
                    dev_loss_evaluator.print_report(args, reset=True, log_to_wandb=log_to_wandb)
            dist.barrier()
            if rank == 0: print(f'Running Train Epoch {epoch}')        
            ddp_model.train()  # Turn dropout back on
            for i, batch in enumerate(train_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, batch_subject_indices, batch_trial_indices = batch

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = ddp_model(inputs)

                # Compute the loss 
                compute_report = i % 100 == 0
                loss = train_loss_evaluator(inputs,
                                            outputs,
                                            labels,
                                            batch_subject_indices,
                                            batch_trial_indices,
                                            args,
                                            compute_report,
                                            log_reports_to_wandb=log_to_wandb)
                if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                    logging.info(f'  - [{rank=}] Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))

                if (i + 1) % 1000 == 0 or i == len(train_dataloader) - 1:
                    logging.info(f'[{rank=}] Batch {i} Training Set Evaluation:')
                    train_loss_evaluator.print_report(args, reset=False)

                    if rank == 0:
                        model_path = f"{checkpoint_dir}/epoch_{epoch}_batch_{i}.pt"
                        if not os.path.exists(os.path.dirname(model_path)):
                            os.makedirs(os.path.dirname(model_path)) 
                        torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': ddp_model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                    }, model_path)

                # Backward pass
                loss.backward()

                # Update the model's parameters
                optimizer.step()

            # Report training loss on this epoch
            if rank == 0:
                logging.info(f"{epoch=} / {epochs}")
                logging.info('-' * 80)
                logging.info(f'Epoch {epoch} Training Set Evaluation: ')
                train_loss_evaluator.print_report(args, log_to_wandb=log_to_wandb)
                logging.info('-' * 80)
                

        # Destroy processes
        wandb.finish()
        dist.destroy_process_group()
        return True
    
    def diffusion_process(self, batch, split, iteration):
        data_len = self.train_data_len if split == 'train' else self.dev_data_len
        for i in range(0, self.batch_size):
            t, weights = self.schedule_sampler.sample(self.batch_size, self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                batch,  # [bs, ch, image_size, image_size]
                t,
                self.device  # [bs](int) sampled timesteps
            )
            # with self.ddp_model.no_sync():
            losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            self.log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}, split, self.log_to_wandb
            )
            if (iteration + 1) % 1000 == 0 or iteration == data_len - 1: #and i == self.batch_size:
                    logging.info(f'[{self.rank=}] Batch {iteration} {split} Set Evaluation:')
                    for key, values in losses.items():
                        logging.info(f' - {key} mean error: {values.mean().item()}')
            if split == 'train':
                loss = (losses["loss"] * weights.to(self.device)).mean()
                loss.backward()

    def log_loss_dict(self, diffusion, ts, losses, split, log_to_wandb=False):
        if not log_to_wandb: pass
        report = {}
        quartile_report = {}
        for key, values in losses.items():
            report[f'{split}/reports/{key}'] = values.mean().item()
            wandb.log(report)
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            quartile_report[f'{split}/reports/{key}/q{quartile}'] = sub_loss
            wandb.log(quartile_report)

    def get_gaussian_diffusion(self, args):
    # default params
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = self.diffusion_steps
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd.get_named_beta_schedule(self.noise_schedule, steps, scale_beta)
        loss_type = gd.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not self.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=self.lambda_vel,
            lambda_pos=self.lambda_pos,
            lambda_acc=self.lambda_acc,
            lambda_fc=self.lambda_fc,
            lambda_wrench=self.lambda_wrench,
            lambda_res_wrench=self.lambda_res_wrench
        )



# python main.py train --model mdm --checkpoint-dir "../checkpoints/init_mdm_test" --opt-type adagrad -- dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --short

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "/n/holyscratch01/pslade_lab/cbrownpinilla/paper_reimplementation/data/addb_dataset" --epochs 500 --short
