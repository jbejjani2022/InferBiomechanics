import torch
from AddBiomechanicsDataset import OutputDataKeys
from typing import Dict
import numpy as np

class RegressionLossEvaluator:
    num_evaluations: int
    sum_loss: float
    sum_timesteps: int
    sum_correct_foot_classifications: float
    sum_com_acc_squared_error: np.ndarray
    sum_contact_forces_squared_error: np.ndarray
    confusion_matrix: np.ndarray

    def __init__(self, contact_forces_weight=1.0):
        self.contact_forces_criterion = torch.nn.MSELoss()
        self.contact_forces_weight = contact_forces_weight

        self.num_evaluations = 0
        self.sum_loss = 0.0
        self.sum_timesteps = 0
        self.sum_contact_forces_N_error = np.zeros(6)
        self.forces = []

    def __call__(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the loss
        loss = self.contact_forces_weight * torch.mean((outputs[OutputDataKeys.CONTACT_FORCES] - labels[OutputDataKeys.CONTACT_FORCES]) ** 2)

        # Keep track of various performance metrics to report
        with torch.no_grad():
            self.num_evaluations += 1
            timesteps = outputs[OutputDataKeys.CONTACT_FORCES].shape[0]
            self.sum_timesteps += timesteps
            self.sum_loss += loss.item()
            self.sum_contact_forces_N_error += torch.sum((outputs[OutputDataKeys.CONTACT_FORCES] - labels[OutputDataKeys.CONTACT_FORCES]) ** 2, dim=0).numpy()
            self.forces.append(labels[OutputDataKeys.CONTACT_FORCES].numpy())
        return loss

    def print_report(self):
        print('\tLoss: ' + str(self.sum_loss / self.num_evaluations))
        self.forces = np.abs(np.concatenate(self.forces))
        print('\tContact force avg N error (per axis), foot 1: ' +
              str(np.sqrt(self.sum_contact_forces_N_error[:3] / self.sum_timesteps)), f"max={np.max(self.forces, axis=0)}, min={np.min(self.forces, axis=0)}, mean={np.mean(self.forces, axis=0)}")
        print('\tContact force avg N error (per axis), foot 2: ' +
              str(np.sqrt(self.sum_contact_forces_N_error[3:] / self.sum_timesteps)))

        # Reset
        self.num_evaluations = 0
        self.sum_loss = 0.0
        self.sum_timesteps = 0
        self.sum_correct_foot_classifications = 0.0
        self.sum_com_acc_mpss_error = np.zeros(3)
        self.sum_contact_forces_N_error = np.zeros(6)
        self.confusion_matrix = np.zeros((4,4), dtype=np.int64)
        self.forces = []
        pass
