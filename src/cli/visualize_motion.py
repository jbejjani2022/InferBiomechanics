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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
        subparser.add_argument('--stride', type=int, default=1,
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
        if 'command' in args and args.command != 'visualize_motion':
            return False
        model_type: str = args.model_type
        checkpoint_dir: str = args.checkpoint_dir
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
            args.dataset_home,
            history_len,
            device=device,
            geometry_folder=geometry,
            testing_with_short_dataset=short,
            output_data_format=output_data_format,
            stride=stride)
        os.environ["MASTER_ADDR"] = 'localhost'
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(rank=0, world_size=1)
        # Create an instance of the model
        model = DDP(self.get_model(23,
                               2,
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
                               device=device))
        self.load_latest_checkpoint(model, checkpoint_dir=checkpoint_dir, map_location='cpu')
        model.eval()

        diffusion = self.get_gaussian_diffusion(args)

        sample_fn = diffusion.p_sample_loop

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8888)

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(0.05)

        frame: int = 0
        batch: int = 0
        playing: bool = True
        num_frames = model.module.num_output_frames

        skel = dev_dataset.skeletons[0]

        print(' - Generating sample...')
        sample = sample_fn(
                    model,
                    # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                    (args.batch_size, model.module.num_output_frames, model.module.output_vector_dim),  # BUG FIX
                    clip_denoised=False,
                    model_kwargs=None,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
        if num_frames == 0:
            print('No frames in dataset!')
            exit(1)
        def onKeyPress(key):
            nonlocal playing
            nonlocal frame
            nonlocal batch
            if key == ' ':
                playing = not playing
            elif key == 'e':
                print(f'Current frame: {frame}')
                frame = (frame + 1) % num_frames

            elif key == 'a':
                frame = (frame - 1) % num_frames

            elif key == 'g':
                batch = (batch + 1) % args.batch_size
                print(f'batch: {batch}')

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame
                nonlocal batch
                nonlocal model
                nonlocal dev_dataset
                nonlocal sample

                positions =  sample[batch][frame][:23]
                velocities = sample[batch][frame][23:46]
                skel.setPositions(positions)
                skel.setVelocities(velocities)
                gui.nativeAPI().renderSkeleton(skel)

                if playing:
                    frame = (frame + 1) % num_frames

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True
    
    def get_gaussian_diffusion(self, args):
    # default params
        predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
        steps = args.diffusion_steps
        scale_beta = 1.  # no scaling
        timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
        learn_sigma = False
        rescale_timesteps = False

        betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
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

