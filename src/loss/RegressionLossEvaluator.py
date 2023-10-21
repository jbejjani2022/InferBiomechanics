import torch
from data.AddBiomechanicsDataset import OutputDataKeys
from typing import Dict
import numpy as np
import logging



class RegressionLossEvaluator:
    num_evaluations: int
    sum_loss: float
    sum_batches: int
    sum_grf_forces_error: float
    sum_grf_forces_percent_error: float
    sum_grf_cop_error: float
    sum_grf_moment_error: float
    sum_grf_wrench_force_error: float
    sum_grf_wrench_moment_error: float
    confusion_matrix: np.ndarray

    def __init__(self, contact_forces_weight=1.0):
        self.contact_forces_criterion = torch.nn.MSELoss()
        self.contact_forces_weight = contact_forces_weight

        self.num_evaluations = 0
        self.sum_batches = 0
        self.sum_grf_forces_error = 0.0
        self.sum_grf_forces_percent_error = 0.0
        self.sum_grf_cop_error = 0.0
        self.sum_grf_moment_error = 0.0
        self.sum_grf_wrench_force_error = 0.0
        self.sum_grf_wrench_moment_error = 0.0
        self.forces = []

    def compute_norms(self, diff: torch.Tensor) -> torch.Tensor:
        diff = diff.view((-1, diff.shape[-2], int(diff.shape[-1] / 3), 3))
        diff = torch.linalg.norm(diff, dim=-1)
        return diff

    def __call__(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Compute the loss
        force_diff = self.compute_norms(outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME])
        force_loss = torch.sum(force_diff ** 2)

        force_norms = self.compute_norms(labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME])
        force_percentage_diff = 100 * force_diff / torch.max(force_norms, torch.ones_like(force_norms) * 5)

        # print('force guess: ' + str(outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :, :]) + ', force truth: ' + str(labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :, :]) + ', force truth difference: ' + str(outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :, :] - labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :, :])+', force diff: ' + str(force_diff[0, :, :]) + ', force percentage diff: ' + str(force_percentage_diff[0, :, :]))

        cop_diff = self.compute_norms(outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME])
        cop_loss = torch.sum(cop_diff ** 2)

        moment_diff = self.compute_norms(outputs[OutputDataKeys.GROUND_CONTACT_MOMENTS_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_MOMENTS_IN_ROOT_FRAME])
        moment_loss = torch.sum(moment_diff ** 2)

        wrench_diff = self.compute_norms(outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME])
        wrench_loss = torch.sum(wrench_diff ** 2)

        assert(force_diff.shape == moment_diff.shape)
        assert(force_diff.shape == cop_diff.shape)

        # Keep track of various performance metrics to report
        with torch.no_grad():
            self.num_evaluations += 1
            timesteps = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
            self.sum_batches += timesteps
            # print(force_diff.mean().item())
            self.sum_grf_forces_error += force_diff.mean().item()
            self.sum_grf_forces_percent_error += force_percentage_diff.mean().item()
            self.sum_grf_cop_error += cop_diff.mean().item()
            self.sum_grf_moment_error += moment_diff.mean().item()
            self.sum_grf_wrench_moment_error += (wrench_diff[:, :, 0].mean().item() + wrench_diff[:, :, 2].mean().item()) / 2
            self.sum_grf_wrench_force_error += (wrench_diff[:, :, 1].mean().item() + wrench_diff[:, :, 3].mean().item()) / 2
        loss = force_loss + cop_loss + moment_loss + wrench_loss
        return loss / timesteps

    def print_report(self):
        print(f'\tForce Avg Err: {self.sum_grf_forces_error / self.num_evaluations} N / kg')
        print(f'\tForce Avg Err: {self.sum_grf_forces_percent_error / self.num_evaluations} %')
        print(f'\tCoP Avg Err: {self.sum_grf_cop_error / self.num_evaluations} m')
        print(f'\tMoment Avg Err: {self.sum_grf_moment_error / self.num_evaluations} Nm / kg')
        print(f'\tWrench Force Avg Err: {self.sum_grf_wrench_force_error / self.num_evaluations} N / kg')
        print(f'\tWrench Moment Avg Err: {self.sum_grf_wrench_moment_error / self.num_evaluations} Nm / kg')

        # Reset
        self.num_evaluations = 0
        # self.sum_loss = 0.0
        self.sum_batches = 0
        self.sum_grf_forces_error = 0.0
        self.sum_grf_forces_percent_error = 0.0
        self.sum_grf_cop_error = 0.0
        self.sum_grf_moment_error = 0.0
        self.sum_grf_wrench_force_error = 0.0
        self.sum_grf_wrench_moment_error = 0.0
        self.forces = []
        pass
