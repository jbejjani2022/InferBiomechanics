import nimblephysics as nimble
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import os
import numpy as np


class InputDataKeys:
    POS = 'pos'
    VEL = 'vel'
    ACC = 'acc'
    COM_ACC = 'com_acc'


class OutputDataKeys:
    CONTACT = 'contact'
    COM_ACC = 'com_acc'
    CONTACT_FORCES = 'contact_forces'


class AddBiomechanicsDataset(Dataset):
    data_path: str
    window_size: int
    stride: int
    device: torch.device
    subjects: List[nimble.biomechanics.SubjectOnDisk]
    input_dofs: List[str]
    input_dof_indices: List[int]
    windows: List[Tuple[nimble.biomechanics.SubjectOnDisk, int, int, str]]

    def __init__(self, subject_paths: List[str], window_size: int, stride: int, input_dofs: List[str], device: torch.device = torch.device('cpu')):
        self.subject_paths = subject_paths
        self.window_size = window_size
        self.stride = stride
        self.input_dofs = input_dofs
        self.device = device
        self.subjects = []
        self.windows = []
            
        for subject_path in subject_paths:
            # Create a subject object for each file. This will load just the header from this file, and keep that around in memory
            subject = nimble.biomechanics.SubjectOnDisk(
                subject_path)
            # Add the subject to the list of subjects
            self.subjects.append(subject)
        
        # Read the dofs from the first subject (assuming they are all the same)
        self.input_dof_indices = []
        skel = self.subjects[0].readSkel()
        dof_names = []
        for i in range(skel.getNumDofs()):
            dof_name = skel.getDofByIndex(i).getName()
            dof_names.append(dof_name)

        for dof_name in input_dofs:
            index = dof_names.index(dof_name)
            if index >= 0:
                self.input_dof_indices.append(index)
            else:
                # Throw an exception
                raise Exception('Dof ' + dof_name + ' not found in input dofs')

        index = 0
        num_skipped = 0   
        for subject in self.subjects:
            # Also, count how many random windows we could select from this subject
            for trial in range(subject.getNumTrials()):
                probably_missing: List[bool] = subject.getProbablyMissingGRF(trial)

                trial_length = subject.getTrialLength(trial)
                all_frames: List[nimble.biomechanics.Frame] = subject.readFrames(trial, 0, numFramesToRead=trial_length // self.stride, stride=self.stride, contactThreshold=0.1)
                for window_start in range(max(len(all_frames) - (window_size) + 1, 0)):
                    # Check if any of the frames in this window are probably missing GRF data
                    # If so, skip this window
                    skip = False
                    for i in range(window_start, window_start + window_size):
                        if probably_missing[i]:
                            skip = True
                            break
                    if not skip:
                        np.random.seed(index)
                        frames = all_frames[window_start:window_start+window_size]
                        # print(f"{len(frames)=}")
                        # We first assemble the data into numpy arrays, and then convert to tensors, to save from spurious memory copies which slow down data loading
                        numpy_input_dict: Dict[str, np.ndarray] = {}
                        numpy_output_dict: Dict[str, np.ndarray] = {}

                        numpy_input_dict[InputDataKeys.POS] = np.column_stack([frame.pos[self.input_dof_indices] for frame in frames])
                        numpy_input_dict[InputDataKeys.VEL] = np.column_stack([frame.vel[self.input_dof_indices] for frame in frames])
                        numpy_input_dict[InputDataKeys.ACC] = np.column_stack([frame.acc[self.input_dof_indices] for frame in frames])
                        numpy_input_dict[InputDataKeys.COM_ACC] = np.column_stack([frame.comAcc for frame in frames])

                        numpy_output_dict[OutputDataKeys.CONTACT] = np.column_stack([np.array(frame.contact, dtype=np.float64) for frame in frames])
                        correct = subject.getContactBodies()[0][-1] == 'l'
                        left = 0 if correct else 1
                        right = 1 - left
                        contact_class = 0
                        if frames[-1].contact[left] == 0 and frames[-1].contact[right] == 0:
                            # Flight phase
                            contact_class = 0
                        elif frames[-1].contact[left] == 1 and frames[-1].contact[right] == 0:
                            # Left foot stance
                            contact_class = 1
                        elif frames[-1].contact[left] == 0 and frames[-1].contact[right] == 1:
                            # Right foot stance
                            contact_class = 2
                        elif frames[-1].contact[left] == 1 and frames[-1].contact[right] == 1:
                            # Double stance
                            contact_class = 3
                        one_hot_contact = np.zeros(4, dtype=np.float32)
                        one_hot_contact[contact_class] = 1

                        numpy_output_dict[OutputDataKeys.CONTACT] = one_hot_contact
                        numpy_output_dict[OutputDataKeys.CONTACT_FORCES] = (frames[-1].groundContactForce if correct else frames[-1].groundContactForce[[3,4,5,0,1,2]]) / (np.array([1.,9.8,1.,1.,9.8,1.]) * subject.getMassKg())

                        # Doing things inside torch.no_grad() suppresses warnings and gradient tracking
                        with torch.no_grad():
                            input_dict: Dict[str, torch.Tensor] = {}
                            for key in numpy_input_dict:
                                input_dict[key] = torch.tensor(
                                    numpy_input_dict[key], dtype=torch.float32, device=self.device)

                            label_dict: Dict[str, torch.Tensor] = {}
                            for key in numpy_output_dict:
                                label_dict[key] = torch.tensor(
                                    numpy_output_dict[key], dtype=torch.float32, device=self.device)

                        self.windows.append((input_dict, label_dict))
                        index += 1
                    else:
                        num_skipped += 1

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index: int):
        input_dict, label_dict = self.windows[index]
        
        return input_dict, label_dict

if __name__ == "__main__":
    window_size = 50
    stride = 20
    batch_size = 32
    device = 'cpu'

    # Input dofs to train on
    input_dofs = ['knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']
    data_path = "/Users/rishi/Documents/Academics/stanford/human-body-dynamics/InferBiomechanics/data/processed/standardized/rajagopal_no_arms/data/protected/us-west-2:be72ee5a-acdb-4e07-b288-a55886ca1e3b/data/c1ab/5dd9f9149f8e8064442a852d79e77050a17c772bdc1199cfe088177b9387a657/5dd9f9149f8e8064442a852d79e77050a17c772bdc1199cfe088177b9387a657.bin"
    AddBiomechanicsDataset(data_path, window_size, stride, input_dofs=input_dofs, device=torch.device(device))