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
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np


class VisualizeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('visualize', help='Visualize the performance of a model on dataset.')

        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
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
        subparser.add_argument('--dropout', action='store_true', help='Apply dropout?')
        subparser.add_argument('--dropout-prob', type=float, default=0.5, help='Dropout prob')
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='relu', help='Which activation func?')
        subparser.add_argument('--batch-size', type=int, default=32,
                               help='The batch size to use when training the model.')
        subparser.add_argument('--short', action='store_true',
                               help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)],
                               help='Which wrench components to train.')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'visualize':
            return False
        model_type: str = args.model_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), model_type)
        history_len: int = args.history_len
        device: str = args.device
        short: bool = args.short
        stride: int = args.stride

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        print('## Loading TRAIN set:')
        train_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/train'),
            history_len,
            device=torch.device(device),
            geometry_folder=geometry,
            testing_with_short_dataset=short,
            stride=stride)
        # print('## Loading DEV set:')
        # dev_dataset = AddBiomechanicsDataset(
        #     os.path.abspath('../data/dev'), history_len, device=torch.device(device), geometry_folder=geometry, testing_with_short_dataset=short)

        # Create an instance of the model
        model = self.get_model(args, train_dataset.num_dofs, train_dataset.num_joints, model_type, history_len, device)
        self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir)
        model.eval()

        # Iterate over the entire training dataset
        loss_evaluator = RegressionLossEvaluator(dataset=train_dataset, split='train')

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8080)

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
            0.04)

        frame: int = 0
        playing: bool = True
        num_frames = len(train_dataset)
        if num_frames == 0:
            print('No frames in dataset!')
            exit(1)

        def onKeyPress(key):
            nonlocal playing
            nonlocal frame
            if key == ' ':
                playing = not playing
            elif key == 'e':
                frame += 1
                if frame >= num_frames - 5:
                    frame = 0
            elif key == 'a':
                frame -= 1
                if frame < 0:
                    frame = num_frames - 5
            elif key == 'r':
                loss_evaluator.print_report()

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame
                nonlocal model
                nonlocal train_dataset

                inputs: Dict[str, torch.Tensor]
                labels: Dict[str, torch.Tensor]
                inputs, labels, batch_subject_index, trial_index = train_dataset[frame]
                batch_subject_indices: List[int] = [batch_subject_index]
                batch_trial_indices: List[int] = [trial_index]

                # Add a batch dimension
                for key in inputs:
                    inputs[key] = inputs[key].unsqueeze(0)
                for key in labels:
                    labels[key] = labels[key].unsqueeze(0)

                # Forward pass
                skel_and_contact_bodies = [(train_dataset.skeletons[i], train_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices]
                outputs = model(inputs, skel_and_contact_bodies)
                skel = skel_and_contact_bodies[0][0]
                contact_bodies = skel_and_contact_bodies[0][1]

                loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=True)
                if frame % 100 == 0:
                    print('Results on Frame ' + str(frame) + '/' + str(num_frames))
                    loss_evaluator.print_report(args)

                ground_forces: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
                left_foot_force = ground_forces[0, 0:3]
                right_foot_force = ground_forces[0, 3:6]

                cops: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
                left_foot_cop = cops[0, 0:3]
                right_foot_cop = cops[0, 3:6]

                predicted_forces = (left_foot_force, right_foot_force)
                predicted_cops = (left_foot_cop, right_foot_cop)

                pos_in_root_frame = np.copy(inputs[InputDataKeys.POS][0, 0, :].cpu().numpy())
                pos_in_root_frame[0:6] = 0
                skel.setPositions(pos_in_root_frame)

                gui.nativeAPI().renderSkeleton(skel)

                joint_centers = inputs[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME][0, 0, :].cpu().numpy()
                num_joints = int(len(joint_centers) / 3)
                for j in range(num_joints):
                    gui.nativeAPI().createSphere('joint_' + str(j), [0.05, 0.05, 0.05], joint_centers[j * 3:(j + 1) * 3],
                                                 [1, 0, 0, 1])

                root_lin_vel = inputs[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME][0, 0, 0:3].cpu().numpy()
                gui.nativeAPI().createLine('root_lin_vel', [[0, 0, 0], root_lin_vel], [1, 0, 0, 1])

                root_pos_history = inputs[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME][0, 0, :].cpu().numpy()
                num_history = int(len(root_pos_history) / 3)
                for h in range(num_history):
                    gui.nativeAPI().createSphere('root_pos_history_' + str(h), [0.05, 0.05, 0.05],
                                                 root_pos_history[h * 3:(h + 1) * 3], [0, 1, 0, 1])

                force_cops = labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0, :].cpu().numpy()
                force_fs = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :].cpu().numpy()
                num_forces = int(len(force_cops) / 3)
                force_index = 0
                for f in range(num_forces):
                    if contact_bodies[f] == 'pelvis':
                        continue
                    cop = force_cops[f * 3:(f + 1) * 3]
                    force = force_fs[f * 3:(f + 1) * 3]
                    gui.nativeAPI().createLine('force_' + str(f),
                                               [cop,
                                                cop + force],
                                               [1, 0, 0, 1])

                    predicted_cop = predicted_cops[force_index] # contact_bodies[f].getWorldTransform().translation() #
                    predicted_force = predicted_forces[force_index]
                    gui.nativeAPI().createLine('predicted_force_' + str(f),
                                               [predicted_cop,
                                                predicted_cop + predicted_force],
                                               [0, 0, 1, 1])
                    force_index += 1

                if playing:
                    frame += 1
                    if frame >= num_frames - 5:
                        frame = 0

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True

