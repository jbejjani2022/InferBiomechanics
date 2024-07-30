import argparse

import torch
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from loss.dynamics.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


class VisualizeMotionCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('visualize_motion', help='Visualize the performance of a model on dataset.')

        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--use-diffusion', action='store_true', help='Use diffusion?')
        subparser.add_argument('--output-data-format', type=str, default='all_frames', choices=['all_frames', 'last_frame'], 
                               help='Output for all frames in a window or only the last frame.')
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
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='sigmoid', help='Which activation func?')
        subparser.add_argument('--batch-size', type=int, default=32,
                               help='The batch size to use when training the model.')
        subparser.add_argument('--short', action='store_true',
                               help='Use very short datasets to test without loading a bunch of data.')
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
        root_history_len: int = 10
        hidden_dims: List[int] = args.hidden_dims
        device: str = args.device
        short: bool = args.short
        stride: int = args.stride
        batchnorm: bool = args.batchnorm
        dropout: bool = args.dropout
        output_data_format: str = args.output_data_format
        activation: str = args.activation

        geometry = self.ensure_geometry(args.geometry_folder)

        # Create an instance of the dataset
        # print('## Loading TRAIN set:')
        # train_dataset = AddBiomechanicsDataset(
        #     os.path.abspath('../data/train'),
        #     history_len,
        #     device=torch.device(device),
        #     geometry_folder=geometry,
        #     testing_with_short_dataset=short,
        #     output_data_format=output_data_format,
        #     stride=stride)
        print('## Loading DEV set:')
        dev_dataset = AddBiomechanicsDataset(
            os.path.abspath('../data/dev'),
            history_len,
            device=torch.device(device),
            geometry_folder=geometry,
            testing_with_short_dataset=short,
            output_data_format=output_data_format,
            stride=stride)

        # Create an instance of the model
        model = self.get_model(dev_dataset.num_dofs,
                               dev_dataset.num_joints,
                               model_type,
                               history_len=history_len,
                               stride=stride,
                               hidden_dims=hidden_dims,
                               activation=activation,
                               batchnorm=batchnorm,
                               dropout=dropout,
                               dropout_prob=0.0,
                               root_history_len=root_history_len,
                               output_data_format=output_data_format,
                               device=device)
        self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir)
        model.eval()

        diffusion = self.get_gaussian_diffusion(args)

        sample_fn = diffusion.p_sample_loop

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8080)

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
            0.04)

        frame: int = 0
        playing: bool = True
        num_frames = len(dev_dataset)
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
            # elif key == 'r':
                # loss_evaluator.print_report()

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame
                nonlocal model
                nonlocal dev_dataset

                # inputs: Dict[str, torch.Tensor]
                # labels: Dict[str, torch.Tensor]
                # inputs, labels, batch_subject_index, trial_index = dev_dataset[frame]
                # batch_subject_indices: List[int] = [batch_subject_index]
                # batch_trial_indices: List[int] = [trial_index]

                # # Add a batch dimension
                # for key in inputs:
                #     inputs[key] = inputs[key].unsqueeze(0)
                # for key in labels:
                #     labels[key] = labels[key].unsqueeze(0)

                # # Forward pass
                # skel_and_contact_bodies = [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices]
                # outputs = model(inputs)
                # skel = skel_and_contact_bodies[0][0]
                # contact_bodies = skel_and_contact_bodies[0][1]

                # loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=True)
                # if frame % 100 == 0:
                #     print('Results on Frame ' + str(frame) + '/' + str(num_frames))
                #     loss_evaluator.print_report(args)

                # subject_path = train_dataset.subject_paths[batch_subject_indices[0]]
                # trial_index = batch_trial_indices[0]
                # print('Subject: ' + subject_path + ', trial: ' + str(trial_index))

                # if output_data_format == 'all_frames':
                #     for key in outputs:
                #         outputs[key] = outputs[key][:, -1, :]
                #     for key in labels:
                #         labels[key] = labels[key][:, -1, :]

                # ground_forces: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
                # left_foot_force = ground_forces[0, 0:3]
                # right_foot_force = ground_forces[0, 3:6]

                # cops: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
                # left_foot_cop = cops[0, 0:3]
                # right_foot_cop = cops[0, 3:6]

                # predicted_forces = (left_foot_force, right_foot_force)
                # predicted_cops = (left_foot_cop, right_foot_cop)

                # pos_in_root_frame = np.copy(inputs[InputDataKeys.POS][0, -1, :].cpu().numpy())
                # pos_in_root_frame[0:6] = 0
                # skel.setPositions(pos_in_root_frame)

                # gui.nativeAPI().renderSkeleton(skel)

                # joint_centers = inputs[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME][0, -1, :].cpu().numpy()
                # num_joints = int(len(joint_centers) / 3)
                # for j in range(num_joints):
                #     gui.nativeAPI().createSphere('joint_' + str(j), [0.05, 0.05, 0.05], joint_centers[j * 3:(j + 1) * 3],
                #                                  [1, 0, 0, 1])

                # root_lin_vel = inputs[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME][0, 0, 0:3].cpu().numpy()
                # gui.nativeAPI().createLine('root_lin_vel', [[0, 0, 0], root_lin_vel], [1, 0, 0, 1])

                # root_pos_history = inputs[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME][0, 0, :].cpu().numpy()
                # num_history = int(len(root_pos_history) / 3)
                # for h in range(num_history):
                #     gui.nativeAPI().createSphere('root_pos_history_' + str(h), [0.05, 0.05, 0.05],
                #                                  root_pos_history[h * 3:(h + 1) * 3], [0, 1, 0, 1])

                # force_cops = labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0, :].cpu().numpy()
                # force_fs = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :].cpu().numpy()
                # num_forces = int(len(force_cops) / 3)
                # force_index = 0
                # for f in range(num_forces):
                #     if contact_bodies[f] == 'pelvis':
                #         continue
                #     cop = force_cops[f * 3:(f + 1) * 3]
                #     force = force_fs[f * 3:(f + 1) * 3]
                #     gui.nativeAPI().createLine('force_' + str(f),
                #                                [cop,
                #                                 cop + force],
                #                                [1, 0, 0, 1])

                #     predicted_cop = predicted_cops[force_index] # contact_bodies[f].getWorldTransform().translation() #
                #     predicted_force = predicted_forces[force_index]
                #     gui.nativeAPI().createLine('predicted_force_' + str(f),
                #                                [predicted_cop,
                #                                 predicted_cop + predicted_force],
                #                                [0, 0, 1, 1])
                #     force_index += 1

                sample = sample_fn(
                    model,
                    # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                    (args.batch_size, model.num_output_frames, model.output_vector_dim),  # BUG FIX
                    clip_denoised=False,
                    model_kwargs=None,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                skel = nimble.RajagopalHumanBodyModel().skeleton

                positions =  sample[OutputDataKeys.POS]
                velocities = sample[OutputDataKeys.VEL]
                skel.setPositions(positions)
                skel.setVelocities(velocities)

                if playing:
                    frame += 1
                    if frame >= num_frames - 5:
                        frame = 0

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True
    
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
                    if not args.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            lambda_vel=args.lambda_vel,
            lambda_pos=args.lambda_pos,
            lambda_acc=args.lambda_acc,
            lambda_fc=args.lambda_fc,
        )

