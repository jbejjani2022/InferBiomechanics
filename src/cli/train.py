import argparse

import torch
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

class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--dataset-home', type=str, default='..', help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints', help='The path to a model checkpoint to save during training. Also, starts from the latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=5, help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--hidden-size', type=int, default=512, help='The hidden size to use when constructing the model.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--learning-rate', type=float, default=1e-2, help='The learning rate for weight updates.')
        subparser.add_argument('--epochs', type=int, default=10, help='The number of epochs to run training for.')
        subparser.add_argument('--opt-type', type=str, default='adagrad', help='The optimizer to use when adapting the weights of the model during training.')
        subparser.add_argument('--batch-size', type=int, default=32, help='The batch size to use when training the model.')
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--num-subjects-prefetch', type=int, default=5, help='Number of subjects to fetch all the data for in each iteration.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        opt_type: str = args.opt_type
        checkpoint_dir: str = os.path.abspath(args.checkpoint_dir)
        history_len: int = args.history_len
        hidden_size: int = args.hidden_size
        learning_rate: float = args.learning_rate
        epochs: int = args.epochs
        batch_size: int = args.batch_size
        device: str = args.device
        short: bool = args.short
        dataset_home: str = args.dataset_home

        geometry = self.ensure_geometry(args.geometry_folder)

        logging.info('Initializing wandb...')
        wandb.init(
            # set the wandb project where this run will be logged
            project="shpd1",

            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "model_type": model_type,
                "optimizer_type": opt_type,
                "epochs": epochs,
            }
        )

        # Create an instance of the dataset
        train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(dataset_home, 'dev'))
        logging.info('## Loading datasets with skeletons:')
        train_dataset = AddBiomechanicsDataset(train_dataset_path, history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)
        dev_dataset = AddBiomechanicsDataset(dev_dataset_path, history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)

        # Create an instance of the model
        model = self.get_model(train_dataset.num_dofs, train_dataset.num_joints, model_type, history_len, hidden_size, device, checkpoint_dir_root=checkpoint_dir)

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
            assert(False)

        for epoch in range(epochs):
            # Iterate over the entire training dataset
            np.random.seed(epoch+9999)
            permuted_indices = np.random.permutation(len(train_dataset.subject_paths))
            
            # Iterate over the entire training dataset
            if args.num_subjects_prefetch < 0:
                args.num_subjects_prefetch = len(train_dataset.subject_paths)

            for subject_index in range(0, len(train_dataset.subject_paths), args.num_subjects_prefetch):
                dataset_creation = time.time()
                subset_indices = permuted_indices[subject_index:subject_index+args.num_subjects_prefetch]
                if args.num_subjects_prefetch < len(train_dataset.subject_paths) or epoch == 0:
                    train_dataset.prepare_data_for_subset(subset_indices)
                # Create a DataLoader to load the data in batches
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                dataset_creation = time.time() - dataset_creation
                logging.info(f"Train Subject Index: {subject_index}/{len(train_dataset.subject_paths)} {dataset_creation=}")
            
                loss_evaluator = RegressionLossEvaluator(dataset=train_dataset)
                for i, batch in enumerate(train_dataloader):
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    inputs, labels, batch_subject_indices = batch

                    # Clear the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(inputs, [(train_dataset.skeletons[i], train_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])

                    # Compute the loss
                    compute_report = i % 100 == 0
                    loss = loss_evaluator(inputs,
                                        outputs,
                                        labels,
                                        batch_subject_indices,
                                        compute_report,
                                        log_reports_to_wandb=True)

                    if i % 100 == 0:
                        logging.info('  - Batch ' + str(i) + '/' + str(len(train_dataloader)))
                    if i % 1000 == 0:
                        loss_evaluator.print_report()
                        model_path = f"{checkpoint_dir}/{model_type}/epoch_{epoch}_batch_{i}.pt"
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
                logging.info(f"{epoch=} / {epochs} {subject_index=} / {len(train_dataset.subject_paths)}")
                logging.info('Training Set Evaluation: ')
                loss_evaluator.print_report()

            # At the end of each epoch, evaluate the model on the dev set
            dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset)
            permuted_indices = np.arange(len(dev_dataset.subject_paths))
            for subject_index in range(0, len(dev_dataset.subject_paths), args.num_subjects_prefetch):
                dataset_creation = time.time()
                subset_indices = permuted_indices[subject_index:subject_index+args.num_subjects_prefetch]
                if args.num_subjects_prefetch < len(dev_dataset.subject_paths) or epoch == 0:
                    dev_dataset.prepare_data_for_subset(subset_indices)
                dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
                dataset_creation = time.time() - dataset_creation
                logging.info(f"Dev batch: {subject_index}/{len(dev_dataset.subject_paths)} {dataset_creation=}")
                with torch.no_grad():
                    for i, batch in enumerate(dev_dataloader):
                        if i % 100 == 0:
                            logging.info('  - Dev Subject Index ' + str(i) + '/' + str(len(dev_dataloader)))
                        inputs: Dict[str, torch.Tensor]
                        labels: Dict[str, torch.Tensor]
                        batch_subject_indices: List[int]
                        inputs, labels, batch_subject_indices = batch
                        outputs = model(inputs, [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])
                        loss = dev_loss_evaluator(inputs, outputs, labels, batch_subject_indices)
            # Report dev loss on this epoch
            logging.info('Dev Set Evaluation: ')
            dev_loss_evaluator.print_report()
        return True