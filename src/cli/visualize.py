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

# The window size is the number of frames we want to have as context for our model to make predictions.
window_size = 5
# The number of timesteps to skip between each frame in a given window. Data is currently all sampled at 100 Hz, so
# this means 0.2 seconds between each frame. This times window_size is the total time span of each window, which is
# currently 2.0 seconds.
stride = 1
# The batch size is the number of windows we want to load at once, for parallel training and inference on a GPU
batch_size = 32

device = 'cpu'

# Input dofs to train on
# input_dofs = ['knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']
input_dofs = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l']
num_joints = 12
history_len = 10


class VisualizeCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser('visualize', help='Visualize the outputs of a model checkpoint, on the AddBiomechanics dataset')
        parser.add_argument("--checkpoint_dir", type=str, help="Directory where the model checkpoints are saved.", default="./outputs/models")
        parser.add_argument("--target_file", type=str, help="File to visualize or process.", default="./data/dev/Falisse2017_Formatted_No_Arm_subject_0.b3d")
        parser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        parser.add_argument("--geometry-folder", type=str, help="Path to the Geometry folder with bone mesh data.", default=None)
        parser.add_argument("--playback_speed", type=float, help="The playback speed for the GUI.", default=1.0)

    def featurize_frame(self, frame: nimble.biomechanics.Frame, input_dof_indices: List[int]) -> Dict[str, torch.Tensor]:
        return {
            InputDataKeys.POS: torch.tensor(frame.processingPasses[-1].pos[input_dof_indices], dtype=torch.float32),
            InputDataKeys.VEL: torch.tensor(frame.processingPasses[-1].vel[input_dof_indices], dtype=torch.float32),
            InputDataKeys.ACC: torch.tensor(frame.processingPasses[-1].acc[input_dof_indices], dtype=torch.float32),
            InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME: torch.tensor(frame.processingPasses[-1].jointCentersInRootFrame,
                                                                    dtype=torch.float32),
            InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootLinearVelInRootFrame,
                dtype=torch.float32),
            InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootAngularVelInRootFrame,
                dtype=torch.float32),
            InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootLinearAccInRootFrame,
                dtype=torch.float32),
            InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootAngularAccInRootFrame,
                dtype=torch.float32),
            InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootPosHistoryInRootFrame,
                dtype=torch.float32),
            InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].rootEulerHistoryInRootFrame,
                dtype=torch.float32),
        }

    def predict_frame(self, frames: List[nimble.biomechanics.Frame],
                      skel: nimble.dynamics.Skeleton,
                      input_dof_indices: List[int],
                      model: FeedForwardBaseline) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        with torch.no_grad():
            featurized: List[Dict[str, torch.Tensor]] = [self.featurize_frame(frame, input_dof_indices) for frame in frames]

            stacked_features: Dict[str, torch.Tensor] = {}
            for key in featurized[0]:
                stacked_features[key] = torch.stack([f[key] for f in featurized], dim=0).unsqueeze(0)

            output_dict: Dict[str, torch.Tensor] = model(stacked_features)

            ground_forces: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
            left_foot_force = ground_forces[0, 0, 0:3]
            right_foot_force = ground_forces[0, 0, 3:6]

            cops: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
            left_foot_cop = cops[0, 0, 0:3]
            right_foot_cop = cops[0, 0, 3:6]

            return (left_foot_force, right_foot_force), (left_foot_cop, right_foot_cop)

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'visualize':
            return False
        output_dir = args.output_dir
        target_file = args.target_file
        trial = args.trial
        playback_speed = args.playback_speed
        geometry = args.geometry

        model = self.get_model()
        self.load_latest_checkpoint(model, output_dir=output_dir)

        geometry = self.ensure_geometry(args.geometry_folder)

        file_path = os.path.abspath(target_file)
        print('Reading SubjectOnDisk at ' + file_path + '...')
        subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file_path)
        print('Subject height: ' + str(subject.getHeightM()) + "m")
        print('Subject mass: ' + str(subject.getMassKg()) + "kg")
        print('Subject biological sex: ' + subject.getBiologicalSex())
        contact_bodies = subject.getGroundForceBodies()
        print('Contact bodies: ' + str(contact_bodies))

        num_frames = subject.getTrialLength(trial)
        skel = subject.readSkel(0, geometry)

        print('DOFs:')
        dof_names: List[str] = []
        for i in range(skel.getNumDofs()):
            print(' [' + str(i) + ']: ' + skel.getDofByIndex(i).getName())
            dof_names.append(skel.getDofByIndex(i).getName())

        input_dof_indices: List[int] = []
        for dof_name in input_dofs:
            index = dof_names.index(dof_name)
            if index >= 0:
                input_dof_indices.append(index)
            else:
                # Throw an exception
                raise Exception('Dof ' + dof_name + ' not found in input dofs')

        world = nimble.simulation.World()
        world.addSkeleton(skel)
        world.setGravity([0, -9.81, 0])
        skel.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8080)

        # Animate the knees back and forth
        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
            subject.getTrialTimestep(trial) / playback_speed)

        frame: int = 0
        playing: bool = True

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

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            nonlocal frame
            nonlocal skel
            nonlocal subject

            loaded: List[nimble.biomechanics.Frame] = subject.readFrames(trial, frame, 5, contactThreshold=20)

            predicted_forces, predicted_cops = self.predict_frame(loaded, skel, input_dof_indices, model)

            pos_in_root_frame = np.copy(loaded[0].processingPasses[-1].pos)
            pos_in_root_frame[0:6] = 0
            skel.setPositions(pos_in_root_frame)

            missing_grf = loaded[0].missingGRFReason != nimble.biomechanics.MissingGRFReason.notMissingGRF

            gui.nativeAPI().renderSkeleton(skel, overrideColor=[1, 0, 0, 1] if missing_grf else [0.7, 0.7, 0.7, 1])

            joint_centers = loaded[0].processingPasses[-1].jointCentersInRootFrame
            num_joints = int(len(joint_centers) / 3)
            for j in range(num_joints):
                gui.nativeAPI().createSphere('joint_' + str(j), [0.05, 0.05, 0.05], joint_centers[j * 3:(j + 1) * 3],
                                             [1, 0, 0, 1])

            root_lin_vel = loaded[0].processingPasses[-1].rootLinearAccInRootFrame
            gui.nativeAPI().createLine('root_lin_vel', [[0, 0, 0], root_lin_vel], [1, 0, 0, 1])

            root_pos_history = loaded[0].processingPasses[-1].rootPosHistoryInRootFrame
            num_history = int(len(root_pos_history) / 3)
            for h in range(num_history):
                gui.nativeAPI().createSphere('root_pos_history_' + str(h), [0.05, 0.05, 0.05],
                                             root_pos_history[h * 3:(h + 1) * 3], [0, 1, 0, 1])

            force_cops = loaded[0].processingPasses[-1].groundContactCenterOfPressureInRootFrame
            force_fs = loaded[0].processingPasses[-1].groundContactForceInRootFrame
            num_forces = int(len(force_cops) / 3)
            force_index = 0
            for f in range(num_forces):
                if contact_bodies[f] == 'pelvis':
                    continue
                cop = force_cops[f * 3:(f + 1) * 3]
                force = force_fs[f * 3:(f + 1) * 3] / skel.getMass()
                gui.nativeAPI().createLine('force_' + str(f),
                                           [cop,
                                            cop + force],
                                           [1, 0, 0, 1])

                body = skel.getBodyNode(contact_bodies[f])
                predicted_cop = body.getWorldTransform().translation()
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
