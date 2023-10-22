import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List
import numpy as np
import wandb


class RegressionLossEvaluator:
    dataset: AddBiomechanicsDataset
    num_evaluations: int
    sum_loss: float
    sum_batches: int
    sum_grf_forces_error: float
    sum_grf_forces_percent_error: float
    sum_grf_cop_error: float
    sum_grf_moment_error: float
    sum_grf_wrench_force_error: float
    sum_grf_wrench_moment_error: float

    def __init__(self, dataset: AddBiomechanicsDataset):
        self.dataset = dataset
        self.num_evaluations = 0
        self.sum_batches = 0
        self.sum_grf_forces_error = 0.0
        self.sum_grf_forces_percent_error = 0.0
        self.sum_grf_cop_error = 0.0
        self.sum_grf_moment_error = 0.0
        self.sum_grf_wrench_force_error = 0.0
        self.sum_grf_wrench_moment_error = 0.0
        self.sum_id_tau_error = 0.0
        self.sum_direct_tau_error = 0.0

    @staticmethod
    def compute_norms(diff: torch.Tensor, max: float = 1e3) -> torch.Tensor:
        diff = diff.view((-1, diff.shape[-2], int(diff.shape[-1] / 3), 3))
        diff = torch.linalg.norm(diff, dim=-1)
        diff = torch.clamp(diff, min=0, max=max)
        return diff

    def __call__(self,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor],
                 labels: Dict[str, torch.Tensor],
                 batch_subject_indices: List[int],
                 compute_report: bool = False,
                 log_reports_to_wandb: bool = False) -> torch.Tensor:
        # Compute the loss
        force_diff = RegressionLossEvaluator.compute_norms(
            outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME])
        force_loss = torch.sum(force_diff ** 2)

        # CoP loss is tricky, because when there is no force the CoP is meaningless, and so we want to ensure that
        # we only report CoP loss on the frames where there is a non-zero force.
        mask_tensor = (labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] != 0).float()
        cop_diff = RegressionLossEvaluator.compute_norms(
            (outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME]) * mask_tensor, max=10.0)
        cop_loss = torch.sum(cop_diff ** 2)

        moment_diff = RegressionLossEvaluator.compute_norms(
            (outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]) * mask_tensor)
        moment_loss = torch.sum(moment_diff ** 2)

        wrench_diff = RegressionLossEvaluator.compute_norms(
            outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME])
        wrench_loss = torch.sum(wrench_diff ** 2)

        assert (force_diff.shape == moment_diff.shape)
        assert (force_diff.shape == cop_diff.shape)

        # Keep track of various performance metrics to report
        if compute_report:
            with torch.no_grad():
                force_norms = RegressionLossEvaluator.compute_norms(
                    labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME])
                force_percentage_diff = 100 * force_diff / torch.max(force_norms, torch.ones_like(force_norms) * 5)

                num_batches = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                num_timesteps = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[1]
                tau_err_mean = 0.0
                for batch in range(num_batches):
                    for timestep in range(num_timesteps):
                        skel = self.dataset.skeletons[batch_subject_indices[batch]]
                        skel.setPositions(inputs[InputDataKeys.POS][batch, timestep, :].cpu().numpy())
                        skel.setVelocities(inputs[InputDataKeys.VEL][batch, timestep, :].cpu().numpy())
                        acc = inputs[InputDataKeys.ACC][batch, timestep, :].cpu().numpy()
                        contact_bodies = self.dataset.skeletons_contact_bodies[batch_subject_indices[batch]]
                        contact_wrench_guesses = outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][batch,
                                                 timestep, :].cpu().numpy() * skel.getMass()
                        contact_wrench_guesses_list = [contact_wrench_guesses[i * 6:i * 6 + 6] for i in
                                                       range(len(contact_bodies))]
                        tau = skel.getInverseDynamicsFromPredictions(acc, contact_bodies, contact_wrench_guesses_list,
                                                                     np.zeros(6))
                        tau_error = tau - labels[OutputDataKeys.TAU][batch, timestep, :].cpu().numpy()
                        # Exclude root residual from error
                        tau_err_mean += np.linalg.norm(tau_error[6:])
                tau_err_mean /= (num_batches * num_timesteps)

                self.num_evaluations += 1
                timesteps = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                self.sum_batches += timesteps
                # print(force_diff.mean().item())
                self.sum_id_tau_error += tau_err_mean
                self.sum_grf_forces_error += force_diff.mean().item()
                self.sum_grf_forces_percent_error += force_percentage_diff.mean().item()
                self.sum_grf_cop_error += cop_diff.mean().item()
                self.sum_grf_moment_error += moment_diff.mean().item()
                self.sum_grf_wrench_moment_error += (wrench_diff[:, :, 0].mean().item() + wrench_diff[:, :,
                                                                                          2].mean().item()) / 2
                self.sum_grf_wrench_force_error += (wrench_diff[:, :, 1].mean().item() + wrench_diff[:, :,
                                                                                         3].mean().item()) / 2

        loss = force_loss + cop_loss + moment_loss + wrench_loss

        if log_reports_to_wandb:
            report: Dict[str, float] = {
                'force_loss': force_loss.item(),
                'cop_loss': cop_loss.item(),
                'moment_loss': moment_loss.item(),
                'wrench_loss': wrench_loss.item(),
                'loss': loss.item()
            }
            if compute_report:
                report['Force Avg Err (N/kg)'] = force_diff.mean().item()
                report['Force Avg Err (%)'] = force_percentage_diff.mean().item()
                report['CoP Avg Err (m)'] = cop_diff.mean().item()
                report['Moment Avg Err (Nm/kg)'] = moment_diff.mean().item()
                report['Wrench Force Avg Err (N/kg)'] = (wrench_diff[:, :, 1].mean().item() + wrench_diff[:, :,
                                                                                       3].mean().item()) / 2
                report['Wrench Moment Avg Err (Nm/kg)'] = (wrench_diff[:, :, 0].mean().item() + wrench_diff[:, :,
                                                                                        2].mean().item()) / 2
                report['Non-root Joint Torques (Inverse Dynamics) Avg Err (Nm/kg)'] = tau_err_mean
            wandb.log(report)

        return loss

    def print_report(self, reset: bool = True, log_to_wandb: bool = False):
        if self.num_evaluations == 0:
            return

        print(f'\tForce Avg Err: {self.sum_grf_forces_error / self.num_evaluations} N / kg')
        print(f'\tForce Avg Err: {self.sum_grf_forces_percent_error / self.num_evaluations} %')
        print(f'\tCoP Avg Err: {self.sum_grf_cop_error / self.num_evaluations} m')
        print(f'\tMoment Avg Err: {self.sum_grf_moment_error / self.num_evaluations} Nm / kg')
        print(f'\tWrench Force Avg Err: {self.sum_grf_wrench_force_error / self.num_evaluations} N / kg')
        print(f'\tWrench Moment Avg Err: {self.sum_grf_wrench_moment_error / self.num_evaluations} Nm / kg')
        print(f'\tNon-root Joint Torques (Inverse Dynamics) Avg Err: {self.sum_id_tau_error / self.num_evaluations} Nm')

        # Reset
        if reset:
            self.num_evaluations = 0
            # self.sum_loss = 0.0
            self.sum_batches = 0
            self.sum_grf_forces_error = 0.0
            self.sum_grf_forces_percent_error = 0.0
            self.sum_grf_cop_error = 0.0
            self.sum_grf_moment_error = 0.0
            self.sum_grf_wrench_force_error = 0.0
            self.sum_grf_wrench_moment_error = 0.0
            self.sum_id_tau_error = 0.0
