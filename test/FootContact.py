import torch
import nimblephysics as nimble

from typing import Dict, List, Tuple

from src.data.AddBiomechanicsDataset import OutputDataKeys, AddBiomechanicsDataset


class AddBiomechanicsDatasetFootContact(AddBiomechanicsDataset):
    def __init__(self,
                 data_path: str,
                 window_size: int,
                 geometry_folder: str,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32,
                 testing_with_short_dataset: bool = False,
                 stride: int = 1,
                 output_data_format: str = 'last_frame',
                 skip_loading_skeletons: bool = False):
        super().__init__(data_path, window_size, geometry_folder, device, dtype,
                         testing_with_short_dataset, stride, output_data_format, skip_loading_skeletons)
        
        self.CONTACT = 'contact'
        self.features = ['ankle_angles', 'subtalar_angles', 'mtp_angles']
        # indices of relevant foot joints in the POS tensor of a given frame, in order of left, right
        self.dof_index = {'ankle_angles': [17, 10],
                          'subtalar_angles': [18, 11],
                          'mtp_angles': [19, 12]}
        
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int]:
        subject_index, trial, window_start = self.windows[index]

        # Read the frames from disk
        subject = self.subjects[subject_index]
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   self.window_size // self.stride,
                                                                   stride=self.stride,
                                                                   includeSensorData=False,
                                                                   includeProcessingPasses=True)
        assert (len(frames) == self.window_size // self.stride)

        first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
        output_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[-1] for frame in frames]

        input_dict: Dict[str, torch.Tensor] = {}
        label_dict: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for feature in self.features:
                input_dict[feature] = torch.row_stack([
                    torch.tensor(p.pos[[self.dof_index[feature]]], dtype=self.dtype).detach() for p in first_passes
                ])
            input_dict[self.CONTACT] = torch.row_stack([
                torch.tensor(p.contact, dtype=self.dtype).detach() for p in first_passes
            ])

            # The output dictionary contains a single frame, the last frame in the window if output_data_format is 2d
            # else it contains outputs for all the frames in first_passes
            start_index = 0 if self.output_data_format == 'all_frames' else -1
            label_dict[OutputDataKeys.TAU] = torch.row_stack([
                torch.tensor(p.tau, dtype=self.dtype).detach() for p in output_passes[start_index:]
            ])
                    
        return input_dict, label_dict, subject_index, trial
