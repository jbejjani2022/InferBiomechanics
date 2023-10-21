import torch
from torch.utils.data import DataLoader
from main import get_model
from AddBiomechanicsDataset import AddBiomechanicsDataset
from LossEvaluator import LossEvaluator
from typing import Dict, Tuple, List
import glob
import pickle

import warnings
warnings.filterwarnings("ignore")

window_size = 50
stride = 20
batch_size = 32
device = 'cpu'

# Input dofs to train on
input_dofs = ['knee_angle_l', 'knee_angle_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_adduction_l', 'hip_adduction_r']

# load trained model
model = get_model()
load_epoch = 0
load_batch = 88000
model_path = f"./outputs/models/epoch_{load_epoch}_batch_{load_batch}.pt"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model_state_dict"])

# analyze a given file
def analyse_file(file_path):
    analyse_dataset = AddBiomechanicsDataset(file_path, window_size, stride, input_dofs=input_dofs, device=torch.device(device))
    analyse_dataloader = DataLoader(analyse_dataset, batch_size=batch_size, shuffle=False)

    analysis_evaluator = LossEvaluator(contact_weight=1.0, com_acc_weight=1e-3, contact_forces_weight=1e-3)

    with torch.no_grad():
        for i, batch in enumerate(analyse_dataloader):
            if i % 100 == 0:
                print('  - Dev Batch ' + str(i) + '/' + str(len(analyse_dataloader)))
            inputs: Dict[str, torch.Tensor]
            labels: Dict[str, torch.Tensor]
            inputs, labels = batch
            outputs = model(inputs)
            loss = analysis_evaluator(outputs, labels)
    return analysis_evaluator

def analyse_folder(folder_path):
    files = glob.glob(f"{folder_path}/**/*.bin", recursive=True)
    for i, file in enumerate(files):
        analysis_evaluator = analyse_file(file)
        pickle.dump((file, analysis_evaluator), open(f"./outputs/analysis/{i}.pkl", "wb"))

if __name__ == "__main__":
    # file_path = "/Users/rishi/Documents/Academics/stanford/human-body-dynamics/InferBiomechanics/data/processed/standardized/rajagopal_no_arms/data/protected/us-west-2:43f17b51-2473-445e-8701-feae8881071f/data/S02/4af1b16b78e1fb1a36964be976ad5bb530b1c9f9e9302a04b5d96282a6d80876/4af1b16b78e1fb1a36964be976ad5bb530b1c9f9e9302a04b5d96282a6d80876.bin"
    # analyse_file(file_path)
    folder_path = "/Users/rishi/Documents/Academics/stanford/human-body-dynamics/InferBiomechanics/data/processed"
    analyse_folder(folder_path)