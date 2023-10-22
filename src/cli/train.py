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


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints', help='The path to a model checkpoint to save during training. Also, starts from the latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=5, help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--hidden-size', type=int, default=512, help='The hidden size to use when constructing the model.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--batch-size', type=int, default=32, help='The batch size to use when training the model.')
        subparser.add_argument('--short', type=bool, default=False, help='Use very short datasets to test without loading a bunch of data.')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        model_type: str = args.model_type
        checkpoint_dir: str = os.path.abspath(args.checkpoint_dir)
        history_len: int = args.history_len
        hidden_size: int = args.hidden_size
        batch_size: int = args.batch_size
        device: str = args.device
        short: bool = args.short

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        print('## Loading TRAIN set:')
        train_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/train'), history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)
        print('## Loading DEV set:')
        dev_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/dev'), history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)

        # Create an instance of the model
        model = self.get_model(train_dataset.num_dofs, train_dataset.num_joints, model_type, history_len, hidden_size, device, checkpoint_dir=checkpoint_dir)

        # Create a DataLoader to load the data in batches
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        # The number of epochs is the number of times we want to iterate over the entire dataset during training
        epochs = 100000
        # Learning rate
        # learning_rate = 1e-3
        learning_rate = 1e-2

        # Define the optimizer
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            # Iterate over the entire training dataset
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
                loss = loss_evaluator(inputs, outputs, labels, batch_subject_indices, compute_report)

                if i % 100 == 0:
                    print('  - Batch ' + str(i) + '/' + str(len(train_dataloader)))
                if i % 1000 == 0:
                    loss_evaluator.print_report()
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
            print('Epoch ' + str(epoch) + ': ')
            print('Training Set Evaluation: ')
            loss_evaluator.print_report()

            # At the end of each epoch, evaluate the model on the dev set
            dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset)
            with torch.no_grad():
                for i, batch in enumerate(dev_dataloader):
                    if i % 100 == 0:
                        print('  - Dev Batch ' + str(i) + '/' + str(len(dev_dataloader)))
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    inputs, labels, batch_subject_indices = batch
                    outputs = model(inputs, [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices])
                    loss = dev_loss_evaluator(inputs, outputs, labels, batch_subject_indices)
            # Report dev loss on this epoch
            print('Dev Set Evaluation: ')
            dev_loss_evaluator.print_report()
        return True