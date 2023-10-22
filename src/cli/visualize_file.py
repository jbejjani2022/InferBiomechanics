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


class VisualizeFileCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser('visualize-file', help='Visualize the outputs of a model checkpoint on a specific B3D file')
        parser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        parser.add_argument("--checkpoint-dir", type=str, help="Directory where the model checkpoints are saved.", default="../checkpoints")
        parser.add_argument("--target-file", type=str, help="File to visualize or process.", default="../data/dev/Falisse2017_Formatted_No_Arm_subject_0.b3d")
        parser.add_argument("--trial", type=int, help="Trial to visualize or process.", default=0)
        parser.add_argument("--geometry-folder", type=str, help="Path to the Geometry folder with bone mesh data.", default=None)
        parser.add_argument("--playback_speed", type=float, help="The playback speed for the GUI.", default=1.0)
        parser.add_argument('--history-len', type=int, default=5, help='The number of timesteps of context to show when constructing the inputs. MUST MATCH THE train COMMAND VALUE!')
        parser.add_argument('--hidden-size', type=int, default=512, help='The hidden size to use when constructing the model. MUST MATCH THE train COMMAND VALUE!')

    def featurize_frame(self, frame: nimble.biomechanics.Frame) -> Dict[str, torch.Tensor]:
        return {
            InputDataKeys.POS: torch.tensor(frame.processingPasses[-1].pos, dtype=torch.float32),
            InputDataKeys.VEL: torch.tensor(frame.processingPasses[-1].vel, dtype=torch.float32),
            InputDataKeys.ACC: torch.tensor(frame.processingPasses[-1].acc, dtype=torch.float32),
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

    def predict_frame(self,
                      frames: List[nimble.biomechanics.Frame],
                      model: FeedForwardBaseline,
                      skel: nimble.dynamics.Skeleton,
                      contact_bodies: List[nimble.dynamics.BodyNode]) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        with torch.no_grad():
            featurized: List[Dict[str, torch.Tensor]] = [self.featurize_frame(frame) for frame in frames]

            stacked_features: Dict[str, torch.Tensor] = {}
            for key in featurized[0]:
                stacked_features[key] = torch.stack([f[key] for f in featurized], dim=0).unsqueeze(0)

            output_dict: Dict[str, torch.Tensor] = model(stacked_features, [(skel, contact_bodies)])

            ground_forces: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
            left_foot_force = ground_forces[0, 0, 0:3]
            right_foot_force = ground_forces[0, 0, 3:6]

            cops: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
            left_foot_cop = cops[0, 0, 0:3]
            right_foot_cop = cops[0, 0, 3:6]

            return (left_foot_force, right_foot_force), (left_foot_cop, right_foot_cop)

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'visualize-file':
            return False
        model_type = args.model_type
        checkpoint_dir = args.checkpoint_dir
        target_file = args.target_file
        trial = args.trial
        playback_speed = args.playback_speed
        history_len = args.history_len
        hidden_size = args.hidden_size

        file_path = os.path.abspath(target_file)
        print('Reading SubjectOnDisk at ' + file_path + '...')
        subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file_path)
        print('Subject height: ' + str(subject.getHeightM()) + "m")
        print('Subject mass: ' + str(subject.getMassKg()) + "kg")
        print('Subject biological sex: ' + subject.getBiologicalSex())
        contact_bodies = subject.getGroundForceBodies()
        print('Contact bodies: ' + str(contact_bodies))

        model = self.get_model(subject.getNumDofs(), subject.getNumJoints(), model_type, history_len, hidden_size, device='cpu', checkpoint_dir=checkpoint_dir)

        geometry = self.ensure_geometry(args.geometry_folder)

        num_frames = subject.getTrialLength(trial)
        skel = subject.readSkel(0, geometry)
        skeleton_contact_bodies = [skel.getBodyNode(name) for name in contact_bodies]

        print('DOFs:')
        dof_names: List[str] = []
        for i in range(skel.getNumDofs()):
            print(' [' + str(i) + ']: ' + skel.getDofByIndex(i).getName())
            dof_names.append(skel.getDofByIndex(i).getName())

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

            predicted_forces, predicted_cops = self.predict_frame(loaded, model, skel, skeleton_contact_bodies)

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
                predicted_cop = predicted_cops[force_index] # body.getWorldTransform().translation()
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
