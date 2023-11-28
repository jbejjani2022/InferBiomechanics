import time

from models.FeedForwardRegressionBaseline import FeedForwardBaseline
import torch
import os
import argparse
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
from typing import List, Dict, Tuple
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
from cli.abstract_command import AbstractCommand


class SavePredictionCSVCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('save-prediction-csv', help='Save the outputs of a model checkpoint on a specific B3D file as a CSV, which can then be loaded in Blender to make high quality renders.')
        subparser.add_argument("--target-file", type=str, help="File to visualize or process.", default="../data/dev/Falisse2017_Formatted_No_Arm_subject_0.b3d")
        subparser.add_argument("--trials", type=int, nargs='+', help="Trials to visualize or process.", default=[22, 23])
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
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
        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)],
                               help='Which wrench components to train.')

    def featurize_frames(self, frame: List[nimble.biomechanics.Frame]) -> Dict[str, torch.Tensor]:
        first_passes = [f.processingPasses[0] for f in frame]
        input_dict: Dict[str, torch.Tensor] = {}
        dtype = torch.float32
        input_dict[InputDataKeys.POS] = torch.row_stack([
            torch.tensor(p.pos, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.VEL] = torch.row_stack([
            torch.tensor(p.vel, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ACC] = torch.row_stack([
            torch.tensor(p.acc, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.jointCentersInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootLinearVelInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootLinearAccInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootAngularVelInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootAngularAccInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootPosHistoryInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        input_dict[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME] = torch.row_stack([
            torch.tensor(p.rootEulerHistoryInRootFrame, dtype=dtype).detach() for p in first_passes
        ])
        return input_dict

    def predict_frame(self,
                      frames: List[nimble.biomechanics.Frame],
                      model: FeedForwardBaseline,
                      skel: nimble.dynamics.Skeleton,
                      contact_bodies: List[nimble.dynamics.BodyNode]) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        with torch.no_grad():
            featurized: Dict[str, torch.Tensor] = self.featurize_frames(frames)
            # Add a batch dimension
            for k in featurized:
                featurized[k] = featurized[k].unsqueeze(0)
            # Run the model
            output_dict: Dict[str, torch.Tensor] = model(featurized)

            ground_forces: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
            left_foot_force = ground_forces[0, -1, 0:3]
            right_foot_force = ground_forces[0, -1, 3:6]

            cops: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
            left_foot_cop = cops[0, -1, 0:3]
            right_foot_cop = cops[0, -1, 3:6]

            return (left_foot_force, right_foot_force), (left_foot_cop, right_foot_cop)

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'save-prediction-csv':
            return False
        target_file: str = args.target_file
        trials: List[int] = args.trials
        model_type: str = args.model_type
        checkpoint_dir: str = os.path.join(os.path.abspath(args.checkpoint_dir), model_type)
        history_len: int = args.history_len
        root_history_len: int = 10
        hidden_dims: List[int] = args.hidden_dims
        device: str = args.device
        stride: int = args.stride
        output_data_format: str = args.output_data_format
        activation: str = args.activation
        batchnorm: bool = args.batchnorm
        dropout: bool = args.dropout

        file_path = os.path.abspath(target_file)
        print('Reading SubjectOnDisk at ' + file_path + '...')
        subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file_path)
        print('Subject height: ' + str(subject.getHeightM()) + "m")
        print('Subject mass: ' + str(subject.getMassKg()) + "kg")
        print('Subject biological sex: ' + subject.getBiologicalSex())
        contact_bodies = subject.getGroundForceBodies()
        model_contact_bodies = ['calcn_r', 'calcn_l']
        print('Contact bodies: ' + str(contact_bodies))

        # Create an instance of the model
        model = self.get_model(subject.getNumDofs(), subject.getNumJoints(),
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

        geometry = self.ensure_geometry(args.geometry_folder)

        skel = subject.readSkel(0, geometry)
        skeleton_contact_bodies = [skel.getBodyNode(name) for name in contact_bodies]

        print('DOFs:')
        dof_names: List[str] = []
        for i in range(skel.getNumDofs()):
            print(' [' + str(i) + ']: ' + skel.getDofByIndex(i).getName())
            dof_names.append(skel.getDofByIndex(i).getName())

        with open('predicted_forces.csv', 'w') as f:
            f.write('t,')
            for i in range(len(model_contact_bodies)):
                f.write(model_contact_bodies[i] + '_x1,' + model_contact_bodies[i] + '_y1,' + model_contact_bodies[i] + '_z1,')
                f.write(model_contact_bodies[i] + '_x2,' + model_contact_bodies[i] + '_y2,' + model_contact_bodies[i] + '_z2')
                if i < len(model_contact_bodies) - 1:
                    f.write(',')
            f.write('\n')

            # This is the rotation for Blender's coordinate system
            rotation = np.zeros((3, 3))
            rotation[0, 0] = 1
            rotation[1, 2] = -1
            rotation[2, 1] = 1

            all_frames: List[nimble.biomechanics.Frame] = []
            for trial in trials:
                print('Trial: ' + subject.getTrialName(trial))
                all_frames.extend(subject.readFrames(trial, 0, subject.getTrialLength(trial), stride=1))

            num_frames = len(all_frames)
            print(len(all_frames))
            for frame in range(num_frames - history_len):
                request_frames = history_len // stride
                loaded: List[nimble.biomechanics.Frame] = all_frames[frame:frame + history_len:stride]
                if len(loaded) < request_frames:
                    continue

                predicted_forces, predicted_cops = self.predict_frame(loaded, model, skel, skeleton_contact_bodies)

                pos_in_root_frame = np.copy(loaded[-1].processingPasses[0].pos)
                # pos_in_root_frame[0:6] = 0
                skel.setPositions(pos_in_root_frame)

                root_body = skel.getRootBodyNode()
                root_transform: nimble.math.Isometry3 = root_body.getWorldTransform()

                force_cops = loaded[-1].processingPasses[0].groundContactCenterOfPressure
                force_fs = loaded[-1].processingPasses[0].groundContactForce
                num_forces = int(len(force_cops) / 3)
                for i in range(num_forces):
                    if contact_bodies[i] == 'pelvis':
                        continue
                    cop = force_cops[i * 3:(i + 1) * 3]
                    force = force_fs[i * 3:(i + 1) * 3]

                force_index = 0
                predicted_force_mags: List[float] = [np.linalg.norm(predicted_forces[i]) for i in range(len(predicted_forces))]
                predicted_force_mag_percentiles = [mag / sum(predicted_force_mags) for mag in predicted_force_mags]
                f.write(str(frame + history_len - 1) + ',')
                for b in range(len(model_contact_bodies)):
                    body = skel.getBodyNode(model_contact_bodies[b])
                    predicted_cop = root_transform.multiply(predicted_cops[force_index])
                    body_transform = body.getWorldTransform().translation()
                    predicted_cop = (predicted_cop + body_transform) / 2.0
                    # predicted_cop = body_transform
                    predicted_force = root_transform.rotation() @ predicted_forces[force_index]
                    predicted_force *= skel.getMass()
                    if predicted_force_mag_percentiles[b] < 0.3:
                        predicted_force = np.zeros(3)

                    end = predicted_cop + predicted_force * 0.001

                    predicted_cop = np.matmul(rotation, predicted_cop)
                    end = np.matmul(rotation, end)

                    f.write(str(predicted_cop[0]) + ',' + str(predicted_cop[1]) + ',' + str(predicted_cop[2]) + ',')
                    f.write(str(end[0]) + ',' + str(end[1]) + ',' + str(end[2]))
                    if b < len(model_contact_bodies) - 1:
                        f.write(',')
                    force_index += 1
                f.write('\n')

            return True
