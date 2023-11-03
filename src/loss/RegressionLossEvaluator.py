import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt
import os
import argparse

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

    def __init__(self, dataset: AddBiomechanicsDataset, split: str):
        self.dataset = dataset
        self.split = split
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
        
        # aggregating losses across batches for dev set evaluation        
        self.losses = []
        self.force_losses = []
        self.moment_losses = []
        self.wrench_losses = []
        self.cop_losses = []
        self.tau_errors = []

        self.outputs = []
        self.labels = []

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
                 batch_trial_indices: List[int],
                 args: argparse.Namespace,
                 compute_report: bool = False,
                 log_reports_to_wandb: bool = False,
                 analyze: bool = False,
                 plot_path_root: str = 'outputs/plots') -> torch.Tensor:
        # Compute the loss
        force_diff = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
        force_loss = torch.mean(force_diff ** 2, dim=(0,1))
        self.force_losses.append(force_loss)
        # CoP loss is tricky, because when there is no force the CoP is meaningless, and so we want to ensure that
        # we only report CoP loss on the frames where there is a non-zero force.
        mask_tensor = (labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] != 0).float()
        cop_diff = (outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME]) * mask_tensor
        cop_loss = torch.mean(cop_diff ** 2, dim=(0,1))
        self.cop_losses.append(cop_loss)

        moment_diff = (outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]) * mask_tensor
        moment_loss = torch.mean(moment_diff ** 2, dim=(0,1))
        self.moment_losses.append(moment_loss)

        wrench_diff = outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME] - labels[
                OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME]
        wrench_loss = torch.mean(wrench_diff ** 2, dim=(0,1))
        self.wrench_losses.append(wrench_loss)

        # Keep track of various performance metrics to report
        tau_err_mean = None
        if compute_report:
            with torch.no_grad():
                # force_norms = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
                # force_percentage_diff = 100 * force_diff / torch.max(torch.abs(force_norms), dim=(0,1))

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
                self.tau_errors.append(tau_err_mean)
                self.num_evaluations += 1
                timesteps = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                self.sum_batches += timesteps
                # print(force_diff.mean().item())
                self.sum_id_tau_error += tau_err_mean
                self.sum_grf_forces_error += force_diff.mean().item()
                # self.sum_grf_forces_percent_error += force_percentage_diff.mean().item()
                self.sum_grf_cop_error += cop_diff.mean().item()
                self.sum_grf_moment_error += moment_diff.mean().item()
                self.sum_grf_wrench_moment_error += (wrench_diff[:, :, 0].mean().item() + wrench_diff[:, :,
                                                                                          2].mean().item()) / 2
                self.sum_grf_wrench_force_error += (wrench_diff[:, :, 1].mean().item() + wrench_diff[:, :,
                                                                                         3].mean().item()) / 2

        loss = torch.sum(force_loss[args.predict_grf_components]) + torch.sum(cop_loss[args.predict_cop_components]) + torch.sum(moment_loss[args.predict_moment_components]) + torch.sum(wrench_loss[args.predict_wrench_components])
        self.losses.append(loss)
        if log_reports_to_wandb:
            self.log_to_wandb(args, force_loss, cop_loss, moment_loss, wrench_loss, loss, tau_err_mean, compute_report=compute_report)
        if analyze:
            self.plot_ferror = ((force_diff)**2)[:,-1,:].reshape(-1,6).detach().numpy()
            for i in args.predict_grf_components:
                plt.clf()
                plt.plot(self.plot_ferror[:,i])
                plt.savefig(os.path.join(plot_path_root, f"{os.path.basename(self.dataset.subject_paths[batch_subject_indices[0]])}_{self.dataset.subjects[batch_subject_indices[0]].getTrialName(batch_trial_indices[0])}_grferror{components[i]}.png"))
        return loss
    
    def log_to_wandb(self, args: argparse.Namespace, force_loss: torch.Tensor, cop_loss: torch.Tensor, moment_loss: torch.Tensor, wrench_loss: torch.Tensor, loss: torch.Tensor, tau_err_mean, compute_report: bool = False):
        components = {0: "left-x", 1: "left-y", 2: "left-z", 3: "right-x", 4: "right-y", 5: "right-z"}
        report: Dict[str, float] = {
            **{f'{self.split}/force_loss/{components[i]}': force_loss[i].item() for i in args.predict_grf_components},
            **{f'{self.split}/cop_loss/{components[i]}': cop_loss[i].item() for i in args.predict_cop_components},
            **{f'{self.split}/moment_loss/{components[i]}': moment_loss[i].item() for i in args.predict_moment_components},
            f'{self.split}/wrench_loss': torch.sum(wrench_loss).item(),
            f'{self.split}/loss': loss.item()
        }
        if compute_report:
            report[f'{self.split}/Force Avg Err (N per kg)'] = force_loss[args.predict_grf_components].mean().item()
            # report['Force Avg Err (%)'] = force_percentage_diff.mean().item()
            report[f'{self.split}/CoP Avg Err (m)'] = cop_loss[args.predict_cop_components].mean().item()
            report[f'{self.split}/Moment Avg Err (Nm per kg)'] = moment_loss[args.predict_moment_components].mean().item()
            report[f'{self.split}/Wrench Force Avg Err (N per kg)'] = (wrench_loss[3:6].mean().item() + wrench_loss[9:12].mean().item()) / 2
            report[f'{self.split}/Wrench Moment Avg Err (Nm per kg)'] = (wrench_loss[:3].mean().item() + wrench_loss[6:9].mean().item()) / 2
            report[f'{self.split}/Non-root Joint Torques (Inverse Dynamics) Avg Err (Nm per kg)'] = tau_err_mean
        #print(report)
        wandb.log(report)

    def print_report(self, args: argparse.Namespace, reset: bool = True, log_to_wandb: bool = False, compute_report: bool = False):     
        if log_to_wandb:
            aggregate_force_loss = torch.mean(torch.vstack(self.force_losses), dim=0)
            aggregate_cop_loss = torch.mean(torch.vstack(self.cop_losses), dim=0)
            aggregate_moment_loss = torch.mean(torch.vstack(self.moment_losses), dim=0)
            aggregate_wrench_loss = torch.mean(torch.vstack(self.wrench_losses), dim=0)
            aggregate_loss = torch.mean(torch.hstack(self.losses))
            if compute_report:
                 aggregate_tau_error = np.mean(self.tau_errors)
            else:
                 aggregate_tau_error = None
            self.log_to_wandb(args, aggregate_force_loss, aggregate_cop_loss, aggregate_moment_loss, aggregate_wrench_loss, aggregate_loss, aggregate_tau_error, compute_report=compute_report)
        #print(f'\tForce Avg Err: {self.sum_grf_forces_error / self.num_evaluations} N / kg')
        #print(f'\tForce Avg Err: {self.sum_grf_forces_percent_error / self.num_evaluations} %')
        #print(f'\tCoP Avg Err: {self.sum_grf_cop_error / self.num_evaluations} m')
        #print(f'\tMoment Avg Err: {self.sum_grf_moment_error / self.num_evaluations} Nm / kg')
        #print(f'\tWrench Force Avg Err: {self.sum_grf_wrench_force_error / self.num_evaluations} N / kg')
        #print(f'\tWrench Moment Avg Err: {self.sum_grf_wrench_moment_error / self.num_evaluations} Nm / kg')
        #print(f'\tNon-root Joint Torques (Inverse Dynamics) Avg Err: {self.sum_id_tau_error / self.num_evaluations} Nm')

        # Reset
        if reset:
            self.num_evaluations = 0
            # self.sum_loss = 0.0
            self.force_losses = []
            self.losses = []
            self.moment_losses = []
            self.cop_losses = []
            self.wrench_losses = []
            self.tau_errors = []
            self.sum_batches = 0
            self.sum_grf_forces_error = 0.0
            self.sum_grf_forces_percent_error = 0.0
            self.sum_grf_cop_error = 0.0
            self.sum_grf_moment_error = 0.0
            self.sum_grf_wrench_force_error = 0.0
            self.sum_grf_wrench_moment_error = 0.0
            self.sum_id_tau_error = 0.0
