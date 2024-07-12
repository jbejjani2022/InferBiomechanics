from models.FeedForwardRegressionBaseline import FeedForwardBaseline
from src.loss.dynamics.RegressionLossEvaluator import RegressionLossEvaluator
import torch
import os
import argparse
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import numpy as np
from typing import List, Dict, Tuple
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
from cli.abstract_command import AbstractCommand
from enum import Enum


class ReviewState(Enum):
    GOOD = 1
    BAD = 2
    WIP = 3


class SubjectToReview:
    subject: nimble.biomechanics.SubjectOnDisk
    model: FeedForwardBaseline
    loss: RegressionLossEvaluator
    skel: nimble.dynamics.Skeleton
    contact_bodies: List[nimble.dynamics.BodyNode]
    window_size: int = 5

    # These are segments that are suspicious for one reason or another, either because of unusually high residuals
    # or because the model is having unusually high loss predicting the ground reaction forces
    suspicious_trial_segments: List[Tuple[int, int, int, ReviewState]] = []

    # These are the individual frames that are marked as suspicious by our heuristics
    trial_frames_losses: List[List[float]] = []
    trial_frames_suspicious: List[List[bool]] = []
    trial_frames_already_ignored: List[List[bool]] = []

    def __init__(self,
                 subject: nimble.biomechanics.SubjectOnDisk,
                 model: FeedForwardBaseline,
                 loss: RegressionLossEvaluator,
                 skel: nimble.dynamics.Skeleton,
                 contact_bodies: List[nimble.dynamics.BodyNode]):
        self.subject = subject
        self.model = model
        self.loss = loss
        self.skel = skel
        self.contact_bodies = contact_bodies

    def save_review_results_csv(self, csv_path: str):
        # Write out the (possibly WIP) results of the review to a CSV file
        with open(csv_path, 'w') as f:
            f.write('trial,start_frame,end_frame,review_result\n')
            for i in range(len(self.suspicious_trial_segments)):
                f.write(str(self.suspicious_trial_segments[i][0]) + ',')
                f.write(str(self.suspicious_trial_segments[i][1]) + ',')
                f.write(str(self.suspicious_trial_segments[i][2]) + ',')
                f.write(str(self.suspicious_trial_segments[i][3]) + '\n')

    def read_review_results_csv(self, csv_path: str):
        # Read in the results of a review from a CSV file
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.split(',')
                trial = int(parts[0])
                start_frame = int(parts[1])
                end_frame = int(parts[2])
                review_result = ReviewState(parts[3])
                self.suspicious_trial_segments.append((trial, start_frame, end_frame, review_result))

    def detect_suspicious_segments(self):
        self.trial_frames_losses = []
        self.trial_frames_suspicious = []
        self.trial_frames_already_ignored = []

        for trial in range(self.subject.getNumTrials()):
            frames_suspicious: List[bool] = []
            frames_losses: List[float] = []
            frames_already_ignored: List[bool] = []

            for frame in range(self.subject.getTrialLength(trial) - self.window_size):
                loaded: List[nimble.biomechanics.Frame] = self.subject.readFrames(trial,
                                                                                  frame,
                                                                                  self.window_size,
                                                                                  contactThreshold=20)
                missing_grf: bool = loaded[-1].missingGRFReason != nimble.biomechanics.MissingGRFReason.notMissingGRF
                if missing_grf:
                    frames_already_ignored.append(True)
                    frames_suspicious.append(False)
                    frames_losses.append(0.0)
                    continue

                output_dict, loss = self.__predict_frame(loaded,
                                                         self.model,
                                                         self.skel,
                                                         self.contact_bodies)

                frames_already_ignored.append(False)
                frames_suspicious.append(False)
                frames_losses.append(loss)

                # self.skel.setPositions(loaded[-1].processingPasses[-1].pos)
                #
                # ground_forces: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
                # left_foot_force = ground_forces[0, 0, 0:3]
                # right_foot_force = ground_forces[0, 0, 3:6]
                #
                # cops: np.ndarray = output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
                # left_foot_cop = cops[0, 0, 0:3]
                # right_foot_cop = cops[0, 0, 3:6]

            self.trial_frames_losses.append(frames_losses)
            self.trial_frames_suspicious.append(frames_suspicious)
            self.trial_frames_already_ignored.append(frames_already_ignored)

        # Now that we have the losses for each frame, we can detect frames with unusually high loss
        avg_loss = 0.0
        frames_counted = 0
        for trial in range(len(self.trial_frames_losses)):
            for frame in range(len(self.trial_frames_losses[trial])):
                if self.trial_frames_already_ignored[trial][frame]:
                    continue
                avg_loss += self.trial_frames_losses[trial][frame]
                frames_counted += 1
        avg_loss /= frames_counted

        # Now that we have the average loss, we can detect suspicious frames
        for trial in range(len(self.trial_frames_losses)):
            for frame in range(len(self.trial_frames_losses[trial])):
                if self.trial_frames_already_ignored[trial][frame]:
                    continue
                if self.trial_frames_losses[trial][frame] > avg_loss * 3:
                    self.trial_frames_suspicious[trial][frame] = True

        # Now that we have the suspicious frames, we can detect suspicious segments
        self.suspicious_trial_segments = []
        for trial in range(len(self.trial_frames_losses)):
            start_frame = -1
            for frame in range(len(self.trial_frames_losses[trial])):
                if self.trial_frames_suspicious[trial][frame] and start_frame == -1:
                    start_frame = frame
                elif not self.trial_frames_suspicious[trial][frame] and start_frame != -1:
                    self.suspicious_trial_segments.append((trial, start_frame, frame - 1, ReviewState.WIP))
                    start_frame = -1
            if start_frame != -1:
                self.suspicious_trial_segments.append((trial, start_frame, len(self.trial_frames_losses[trial]) - 1,
                                                       ReviewState.WIP))

    def __featurize_frame(self, frame: nimble.biomechanics.Frame) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        input_dict: Dict[str, torch.Tensor] = {
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

        mass = self.subject.getMassKg()
        ground_contact_wrenches: np.ndarray = np.zeros(6 * len(self.contact_bodies), dtype=np.float32)
        ground_contact_cops: np.ndarray = np.zeros(3 * len(self.contact_bodies), dtype=np.float32)
        ground_contact_torques = np.zeros(3 * len(self.contact_bodies), dtype=np.float32)
        ground_contact_forces = np.zeros(3 * len(self.contact_bodies), dtype=np.float32)
        contact_indices: List[int] = [
            self.subject.getGroundForceBodies().index(
                body.getName()) if body.getName() in self.subject.getGroundForceBodies() else -1 for body in
            self.contact_bodies]
        for i in range(len(self.contact_bodies)):
            if contact_indices[i] >= 0:
                ground_contact_wrenches[6 * i:6 * i + 6] = frame.processingPasses[-1].groundContactWrenchesInRootFrame[
                                                           6 * contact_indices[i]:6 * contact_indices[i] + 6] / mass
                ground_contact_cops[3 * i:3 * i + 3] = frame.processingPasses[
                                                           -1].groundContactCenterOfPressureInRootFrame[
                                                       3 * contact_indices[i]:3 * contact_indices[i] + 3]
                ground_contact_torques[3 * i:3 * i + 3] = frame.processingPasses[-1].groundContactTorqueInRootFrame[
                                                          3 * contact_indices[i]:3 * contact_indices[i] + 3] / mass
                ground_contact_forces[3 * i:3 * i + 3] = frame.processingPasses[-1].groundContactForceInRootFrame[
                                                         3 * contact_indices[i]:3 * contact_indices[i] + 3] / mass

        label_dict: Dict[str, torch.Tensor] = {
            OutputDataKeys.TAU: torch.tensor(frame.processingPasses[-1].tau, dtype=torch.float32),
            OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: torch.tensor(ground_contact_cops, dtype=torch.float32),
            OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: torch.tensor(ground_contact_forces,
                                                                             dtype=torch.float32),
            OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: torch.tensor(ground_contact_torques,
                                                                              dtype=torch.float32),
            OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: torch.tensor(ground_contact_wrenches,
                                                                               dtype=torch.float32),
            OutputDataKeys.RESIDUAL_WRENCH_IN_ROOT_FRAME: torch.tensor(
                frame.processingPasses[-1].residualWrenchInRootFrame, dtype=torch.float32),
            OutputDataKeys.COM_ACC_IN_ROOT_FRAME: torch.tensor(frame.processingPasses[-1].comAccInRootFrame,
                                                               dtype=torch.float32),
        }

        return input_dict, label_dict

    def __predict_frame(self,
                        frames: List[nimble.biomechanics.Frame],
                        model: FeedForwardBaseline,
                        skel: nimble.dynamics.Skeleton,
                        contact_bodies: List[nimble.dynamics.BodyNode]) -> Tuple[Dict[str, torch.Tensor], float]:
        with torch.no_grad():
            featurized: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = [self.__featurize_frame(frame) for
                                                                                         frame in frames]

            stacked_input_features: Dict[str, torch.Tensor] = {}
            stacked_label_features: Dict[str, torch.Tensor] = {}
            for key in featurized[0][0]:
                stacked_input_features[key] = torch.stack([f[0][key] for f in featurized], dim=0).unsqueeze(0)
            for key in featurized[0][1]:
                stacked_label_features[key] = torch.stack([f[1][key] for f in featurized], dim=0).unsqueeze(0)

            output_dict: Dict[str, torch.Tensor] = model(stacked_input_features, [(skel, contact_bodies)])

            loss: torch.Tensor = self.loss(stacked_input_features, output_dict, stacked_label_features, [0], [0])

            return output_dict, loss.item()


class ReviewFileCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser('review-file',
                                       help='Manually review a file for suspicious segments, where the data may not be correct. Visualize the outputs of a model checkpoint on a specific B3D file')
        parser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        parser.add_argument("--checkpoint-dir", type=str, help="Directory where the model checkpoints are saved.",
                            default="../checkpoints")
        parser.add_argument("--target-file", type=str, help="File to visualize or process.",
                            default="../data/dev/Falisse2017_Formatted_No_Arm_subject_0.b3d")
        parser.add_argument("--geometry-folder", type=str, help="Path to the Geometry folder with bone mesh data.",
                            default=None)
        parser.add_argument('--history-len', type=int, default=5,
                            help='The number of timesteps of context to show when constructing the inputs. MUST MATCH THE train COMMAND VALUE!')
        parser.add_argument('--hidden-size', type=int, default=512,
                            help='The hidden size to use when constructing the model. MUST MATCH THE train COMMAND VALUE!')

    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'review-file':
            return False
        model_type = args.model_type
        checkpoint_dir = args.checkpoint_dir
        target_file = args.target_file
        history_len = args.history_len
        hidden_size = args.hidden_size

        file_path = os.path.abspath(target_file)
        print('Reading SubjectOnDisk at ' + file_path + '...')
        subject: nimble.biomechanics.SubjectOnDisk = nimble.biomechanics.SubjectOnDisk(file_path)
        print('Subject height: ' + str(subject.getHeightM()) + "m")
        print('Subject mass: ' + str(subject.getMassKg()) + "kg")
        print('Subject biological sex: ' + subject.getBiologicalSex())
        contact_bodies = [body for body in subject.getGroundForceBodies() if body != 'pelvis']
        print('Contact bodies: ' + str(contact_bodies))

        model = self.get_model(subject.getNumDofs(), subject.getNumJoints(), model_type, history_len, hidden_size,
                               device='cpu', checkpoint_dir=checkpoint_dir)

        geometry = self.ensure_geometry(args.geometry_folder)

        skel = subject.readSkel(0, geometry)
        skeleton_contact_bodies = [skel.getBodyNode(name) for name in contact_bodies]

        to_review: SubjectToReview = SubjectToReview(subject, model, RegressionLossEvaluator(dataset=None), skel, skeleton_contact_bodies)
        print('Predicting suspicious segments...')
        to_review.detect_suspicious_segments()
        print('Got suspicious segments: '+str(to_review.suspicious_trial_segments))

        world = nimble.simulation.World()
        world.addSkeleton(skel)
        world.setGravity([0, -9.81, 0])
        skel.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8080)

        # Animate at 10 FPS
        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(0.1)

        frame: int = 0
        playing: bool = True
        suspicious_segment: int = 0

        def onKeyPress(key):
            nonlocal playing
            nonlocal frame
            nonlocal suspicious_segment
            nonlocal to_review

            trial, start_frame, end_frame, review_state = to_review.suspicious_trial_segments[suspicious_segment]

            if key == ' ':
                playing = not playing
            elif key == 'e':
                frame += 1
                if frame >= end_frame:
                    frame = start_frame
            elif key == 'a':
                frame -= 1
                if frame < start_frame:
                    frame = end_frame
            elif key == 'n':
                suspicious_segment += 1
                if suspicious_segment >= len(to_review.suspicious_trial_segments):
                    suspicious_segment = 0

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            nonlocal frame
            nonlocal suspicious_segment
            nonlocal skel
            nonlocal subject
            nonlocal to_review

            trial, start_frame, end_frame, review_state = to_review.suspicious_trial_segments[suspicious_segment]
            if frame < start_frame or frame > end_frame:
                frame = start_frame

            loaded: List[nimble.biomechanics.Frame] = subject.readFrames(trial, frame, 1, contactThreshold=20)

            skel.setPositions(loaded[0].processingPasses[-1].pos)
            gui.nativeAPI().renderSkeleton(skel)

            force_cops = loaded[0].rawForcePlateCenterOfPressures
            force_fs = loaded[0].rawForcePlateForces
            num_forces = len(force_cops)
            force_index = 0
            for f in range(num_forces):
                cop = force_cops[f]
                force = force_fs[f] / skel.getMass()
                gui.nativeAPI().createLine('force_' + str(f),
                                           [cop,
                                            cop + force],
                                           [1, 0, 0, 1])
                force_index += 1

            if playing:
                frame += 1
                if frame >= end_frame:
                    frame = start_frame

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True
