import os
import wandb
import logging 
import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from cli.utilities import get_git_hash, has_uncommitted_changes
from typing import Dict, List
from cli.abstract_command import AbstractCommand


class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')
        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--no-wandb', action='store_true', default=False,
                               help='Log this run to Weights and Biases.')
        subparser.add_argument('--model-type', type=str, default='feedforward', choices=['analytical', 'feedforward', 'groundlink'], help='The model to train.')
        subparser.add_argument('--output-data-format', type=str, default='all_frames', choices=['all_frames', 'last_frame'], help='Output for all frames in a window or only the last frame.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                               help='The path to a model checkpoint to save during training. Also, starts from the '
                                    'latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None,
                               help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=5,
                               help='The timestep gap between frames in the context window to be used when constructing the inputs.')
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
        subparser.add_argument('--compute-report', action='store_true', default=False,
                               help='Compute inverse dynamics reports during loss evaluation.')

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
        batch_size: int = args.batch_size  # size of batch for any given process
        short: bool = args.short
        dataset_home: str = args.dataset_home
        log_to_wandb: bool = not args.no_wandb
        data_loading_workers: int = args.data_loading_workers
        stride: int = args.stride
        batchnorm: bool = args.batchnorm
        dropout: bool = args.dropout
        dropout_prob: float = args.dropout_prob
        activation: str = args.activation
        compute_report: bool = args.compute_report

        geometry = self.ensure_geometry(args.geometry_folder)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)

        has_uncommitted = has_uncommitted_changes()
        if has_uncommitted:
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error(
                "WARNING: UNCOMMITTED CHANGES IN REPO! THIS WILL MAKE IT HARD TO REPLICATE THIS EXPERIMENT LATER")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            
        if log_to_wandb:
            # Grab all cmd args and add current git hash
            config = args.__dict__
            config["git_hash"] = get_git_hash

            logging.info(f'Initializing wandb...')
            # Check if WANDB_RUN_GROUP environment variable exists
            wandb_group = os.getenv('WANDB_RUN_GROUP', f'ddp_{wandb.util.generate_id()}')  # Default to 'DDP' if not set
            wandb.init(
                # set the wandb project where this run will be logged
                project="addbiomechanics-baseline",

                # track hyperparameters and run metadata
                config=config,
                group=wandb_group
            )

        # Create an instance of the dataset
        DEV = 'test'
        train_dataset_path = os.path.abspath(os.path.join(dataset_home, 'train'))
        dev_dataset_path = os.path.abspath(os.path.join(dataset_home, DEV))

        # Get dataloaders for train and test sets
        print(f"Initializing training set...")
        data_device = device     # device where data will be loaded
        train_dataset = AddBiomechanicsDataset(train_dataset_path, history_len, device=torch.device(data_device), stride=stride, output_data_format=output_data_format,
                                               geometry_folder=geometry, testing_with_short_dataset=short)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True)

        print(f"Initializing dev set...")
        dev_dataset = AddBiomechanicsDataset(dev_dataset_path, history_len, device=torch.device(data_device), stride=stride, output_data_format=output_data_format,
                                             geometry_folder=geometry, testing_with_short_dataset=short)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=data_loading_workers, persistent_workers=True)
        
        # Get loss evaluators
        train_loss_evaluator = RegressionLossEvaluator(dataset=train_dataset, split='train', device=device)
        dev_loss_evaluator = RegressionLossEvaluator(dataset=dev_dataset, split=DEV, device=device)
        
        # Create an instance of the model
        print(f"Initializing model...")
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
                               device=device).to(device)

        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        if not list(params_to_optimize):
            print("No parameters to optimize. Skipping training loop.")
            return False
        
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

        epoch_checkpoint, batch_checkpoint = self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir, optimizer=optimizer)

        for epoch in range(epoch_checkpoint + 1, epochs):
            print(f'Evaluating Dev Set Before Epoch {epoch}')
            with torch.no_grad():
                model.eval()  # Turn dropout off
                for i, batch in enumerate(dev_dataloader):
                    inputs: Dict[str, torch.Tensor]
                    labels: Dict[str, torch.Tensor]
                    batch_subject_indices: List[int]
                    batch_trial_indices: List[int]

                    inputs, labels, batch_subject_indices, batch_trial_indices = batch
                    outputs = model(inputs, i)
                    
                    dev_loss_evaluator(inputs,
                                        outputs,
                                        labels,
                                        batch_subject_indices,
                                        batch_trial_indices,
                                        args,
                                        compute_report=compute_report)

                    if (i + 1) % 100 == 0 or i == len(dev_dataloader) - 1:
                        print(f'  - Dev Batch ' + str(i + 1) + '/' + str(len(dev_dataloader)))
            
                # Report dev loss on this epoch
                print(f'Dev Set Evaluation: ')
                dev_loss_evaluator.print_report(args, log_to_wandb=log_to_wandb)
            
            print(f'Running Training Epoch {epoch}')
            model.train()  # Turn dropout back on
           
            # Iterate over training set
            for i, batch in enumerate(train_dataloader):
                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                batch_subject_indices: List[int]
                batch_trial_indices: List[int]
                inputs, labels, batch_subject_indices, batch_trial_indices = batch

                if i == 0:
                    # print(f'INPUTS dict before reshaping for forward pass: {inputs}')
                    # print(f'LABELS dict: {labels}')
                    # print(f'true GROUND_CONTACT_COPS_IN_ROOT_FRAME: {labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0]}')
                    # print(f'true GROUND_CONTACT_FORCES_IN_ROOT_FRAME: {labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0]}')
                    # print(f'true GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: {labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME][0]}')
                    # print(f'true GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: {labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][0]}')
                    pass

                # Clear the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs, i)
                
                if i == 0:
                    # print(f'OUTPUTS dict: {outputs}')
                    pass
                
                # Compute the loss
                loss = train_loss_evaluator(inputs,
                                            outputs,
                                            labels,
                                            batch_subject_indices,
                                            batch_trial_indices,
                                            args,
                                            compute_report = compute_report and (i % 100 == 0),
                                            log_reports_to_wandb=log_to_wandb)

                if (i + 1) % 100 == 0 or i == len(train_dataloader) - 1:
                    logging.info(f'  - Batch ' + str(i + 1) + '/' + str(len(train_dataloader)))
                
                if (i + 1) % 1000 == 0 or i == len(train_dataloader) - 1:
                    logging.info(f'Batch {i} Training Set Evaluation: ')
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
            logging.info('-' * 80)
            logging.info(f'Epoch {epoch}/{epochs} Training Set Evaluation: ')
            logging.info('-' * 80)
            train_loss_evaluator.print_report(args, log_to_wandb=log_to_wandb)
            logging.info('-' * 80)
            
        return True


# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "../data" --epochs 500

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/checkpoint-gait-ly-only" --hidden-dims 32 32 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adagrad --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --epochs 500 --short

# python3 main.py train --model feedforward --checkpoint-dir "../checkpoints/test" --hidden-dims 256 256 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.001 --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --epochs 5 --short --no-wandb --batch-size 128 --data-loading-workers 1

# Increased batch size to speed up training. Added multiprocessing. Switched from adagrad to adam. Increased hidden layer dim from 32 to 512.
# export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
# export NCCL_DEBUG=INFO
# torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main.py train --model feedforward --checkpoint-dir "../short-feedforward-batchsize-128/checkpoint-gait-ly-only" --hidden-dims 512 512 --batchnorm --dropout --dropout-prob 0.5 --activation tanh --learning-rate 0.01 --opt-type adam --dataset-home "/n/holyscratch01/pslade_lab/AddBiomechanicsDataset/addb_dataset" --epochs 300 --short --batch-size 128