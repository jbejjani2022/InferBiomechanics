import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging
import subprocess


# Utility to get the current repo's git hash, which is useful for replicating runs later
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Git hash could not be found."


# Utility to check if the current repo has uncommited changes, which is useful for debugging why we can't replicate
# runs later, and also yelling at people if they run experiments with uncommited changes.
def has_uncommitted_changes():
    try:
        # The command below checks for changes including untracked files.
        # You can modify this command as per your requirement.
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        return bool(status)
    except subprocess.CalledProcessError:
        return "Could not determine if there are uncommitted changes."


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--no-wandb', action='store_true', default=False,
                               help='Log this run to Weights and Biases.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                               help='The path to a model checkpoint to save during training. Also, starts from the '
                                    'latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None,
                               help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=5,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--learning-rate', type=float, default=1e-2,
                               help='The learning rate for weight updates.')
        subparser.add_argument('--dropout', action='store_true', help='Apply dropout?')
        subparser.add_argument('--dropout-prob', type=float, default=0.5, help='Dropout prob')
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='relu', help='Which activation func?')
        subparser.add_argument('--epochs', type=int, default=10, help='The number of epochs to run training for.')
        subparser.add_argument('--opt-type', type=str, default='adagrad',
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

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        opt_type: str = args.opt_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), model_type)
        history_len: int = args.history_len
        hidden_dims: List[int] = args.hidden_dims
        learning_rate: float = args.learning_rate
        epochs: int = args.epochs
        batch_size: int = args.batch_size
        device: str = args.device
        short: bool = args.short
        dataset_home: str = args.dataset_home
        log_to_wandb: bool = not args.no_wandb
        data_loading_workers: int = args.data_loading_workers

        geometry = self.ensure_geometry(args.geometry_folder)

        has_uncommitted = has_uncommitted_changes()
        if has_uncommitted:
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error(
                "ERROR: UNCOMMITTED CHANGES IN REPO! THIS WILL MAKE IT HARD TO REPLICATE THIS EXPERIMENT LATER")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if log_to_wandb:
            logging.info('Initializing wandb...')
            wandb.init(
                # set the wandb project where this run will be logged
                project="shpd1",

                # track hyperparameters and run metadata
                config=args.__dict__
            )

        # Create an instance of the dataset
        train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(dataset_home, 'dev'))
        logging.info('## Loading datasets with skeletons:')
        train_dataset = AddBiomechanicsDataset(train_dataset_path, history_len, device=torch.device(device), stride=args.stride,
                                               geometry_folder=geometry, testing_with_short_dataset=short)
        train_loss_evaluator = RegressionLossEvaluator(dataset=train_dataset, split='train')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loading_workers, persistent_workers=True)

        dev_dataset = AddBiomechanicsDataset(dev_dataset_path, history_len, device=torch.device(device), stride=args.stride,
                                             geometry_folder=geometry, testing_with_short_dataset=short)
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset, split='dev')
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True)

        mp.set_start_method('spawn')  # 'spawn' or 'fork' or 'forkserver'

        # Create an instance of the model
        model = self.get_model(args, train_dataset.num_dofs, train_dataset.num_joints, model_type, history_len, device)

        # Define the optimizer
        if opt_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif opt_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif opt_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
        elif opt_type == 'adamax':
            optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        else:
            logging.error('Invalid optimizer type: ' + opt_type)
            assert (False)

        self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir, optimizer=optimizer)

        for epoch in range(epochs):
            # Iterate over the entire training dataset

            if epoch > 0:
                print('Evaluating Dev Set')
                with torch.no_grad():
                    for i, batch in enumerate(dev_dataloader):
                        inputs: Dict[str, torch.Tensor]
                        labels: Dict[str, torch.Tensor]
                        batch_subject_indices: List[int]
                        batch_trial_indices: List[int]
                        inputs, labels, batch_subject_indices, batch_trial_indices = batch
                        outputs = model(inputs,
                                        [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in
                                         batch_subject_indices])
                        dev_loss_evaluator(inputs,
                                           outputs,
                                           labels,
                                           batch_subject_indices,
                                           batch_trial_indices,
                                           args,
                                           compute_report=False)
                        if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                            print('  - Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
                # Report dev loss on this epoch
                logging.info('Dev Set Evaluation: ')
                dev_loss_evaluator.print_report(args, reset=True, log_to_wandb=log_to_wandb, compute_report=True)

            print('Running Train Epoch '+str(epoch))
            for i, batch in enumerate(train_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, batch_subject_indices, batch_trial_indices = batch

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs,
                                [(train_dataset.skeletons[i], train_dataset.skeletons_contact_bodies[i]) for i in
                                 batch_subject_indices])

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
                    logging.info('  - Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))
                if (i + 1) % 1000 == 0:
                    train_loss_evaluator.print_report(args, reset=False)
                    model_path = f"{checkpoint_dir}/epoch_{epoch}_batch_{i}.pt"
                    if not os.path.exists(os.path.dirname(model_path)):
                        os.makedirs(os.path.dirname(model_path))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, model_path)

                # Backward pass
                loss.backward()

                # Update the model's parameters
                optimizer.step()
            # Report training loss on this epoch
            logging.info(f"{epoch=} / {epochs}")
            logging.info('Training Set Evaluation: ')
            train_loss_evaluator.print_report(args, log_to_wandb=log_to_wandb)
        return True

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm True --dropout True --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500
