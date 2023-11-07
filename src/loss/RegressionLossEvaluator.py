import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List, Optional
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
    sum_grf_forces_error: float
    sum_grf_forces_percent_error: float
    sum_grf_cop_error: float
    sum_grf_moment_error: float
    sum_grf_wrench_error: float

    def __init__(self, dataset: AddBiomechanicsDataset, split: str):
        self.dataset = dataset
        self.split = split
        self.num_evaluations = 0
        self.sum_grf_forces_error = 0.0
        self.sum_grf_forces_percent_error = 0.0
        self.sum_grf_cop_error = 0.0
        self.sum_grf_moment_error = 0.0
        self.sum_grf_wrench_error = 0.0
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
    def get_squared_diff_mean_vector(output_tensor: torch.Tensor, label_tensor: torch.Tensor) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 2:
            raise ValueError('Output and label tensors must be 2-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] == 0:
            raise ValueError('Output and label tensors must not be empty')
        force_diff = output_tensor - label_tensor
        force_loss = torch.mean(force_diff ** 2, dim=0)
        return force_loss

    @staticmethod
    def get_mask_by_threes(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        with torch.no_grad():
            if len(tensor.shape) != 2:
                raise ValueError('Mask tensor must be 2-dimensional')
            if tensor.shape[0] * tensor.shape[1] == 0:
                raise ValueError('Mask tensor must not be empty')
            if tensor.shape[-1] % 3 != 0:
                raise ValueError('Mask tensor must have a final dimension divisible by 3')

            # Reshape the tensor so that the last dimension is split into chunks of 3
            reshaped_tensor = tensor.view(tensor.shape[0], -1, 3)

            # Compute the norm over the last dimension
            norms = torch.norm(reshaped_tensor, dim=2)

            # Create a mask where the norm is greater than the threshold
            mask = (norms > threshold).float()

            # Expand the mask to cover the original last dimension size
            expanded_mask = mask.unsqueeze(2).expand(-1, -1, 3)
            # print(f"{expanded_mask.shape=}")
            # Reshape the expanded mask back to the original tensor shape
            return expanded_mask.reshape(tensor.shape)

            # Un-vectorized version:

            # mask_tensor = torch.zeros_like(tensor)
            # for batch in range(mask_tensor.shape[0]):
            #     for timestep in range(mask_tensor.shape[1]):
            #         for i in range(mask_tensor.shape[2] // 3):
            #             mask_tensor[batch, timestep, i * 3:(i + 1) * 3] = (torch.norm(tensor[batch, timestep, i * 3:(i + 1) * 3]) > threshold).float()
            # return mask_tensor

    @staticmethod
    def get_mean_norm_error(output_tensor: torch.Tensor, label_tensor: torch.Tensor, vec_size: int = 3) -> torch.Tensor:
        if output_tensor.shape != label_tensor.shape:
            raise ValueError('Output and label tensors must have the same shape')
        if len(output_tensor.shape) != 2:
            raise ValueError('Output and label tensors must be 2-dimensional')
        if output_tensor.shape[0] * output_tensor.shape[1] == 0:
            raise ValueError('Output and label tensors must not be empty')
        if output_tensor.shape[-1] % vec_size != 0:
            raise ValueError('Tensors must have a final dimension divisible by vec_size='+str(vec_size))

        diffs = output_tensor - label_tensor

        # Reshape the tensor so that the last dimension is split into chunks of `vec_size`
        reshaped_tensor = diffs.view(diffs.shape[0], -1, vec_size)

        # Compute the norm over the last dimension
        norms = torch.norm(reshaped_tensor, dim=2)

        # Compute the mean norm over all the dimensions
        mean_norm = torch.mean(norms)

        return mean_norm

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

        ############################################################################
        # Step 1: Compute the loss
        ############################################################################

        # 1.1. Compute the force loss, as a single vector of length 3*N
        force_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
        )
        self.force_losses.append(force_loss)
        # 1.2. Compute the moment loss, as a single vector of length 3*N
        moment_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]
        )
        self.moment_losses.append(moment_loss)
        # 1.3. Compute the wrench loss, as a single vector of length 6*N. Note that this is NOT MASKED the same way the
        # CoP loss is, because unlike the CoP the forces and moments in the wrench are well behaved when the foot is not
        # in contact with the ground: they simply go to zero.
        wrench_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
            labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME]
        )
        self.wrench_losses.append(wrench_loss)
        # 1.4. CoP loss is tricky, because when there is no force the CoP is meaningless, and so we want to ensure that
        # we only report CoP loss on the frames where there is a 10N force or more.
        with torch.no_grad():
            cop_mask_tensor = RegressionLossEvaluator.get_mask_by_threes(
                labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
                threshold=10.0
            )
        cop_loss = RegressionLossEvaluator.get_squared_diff_mean_vector(
            outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor,
            labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor
        )
        self.cop_losses.append(cop_loss)
        # 1.5. We allow the user to specify which axis of the force, moment, and wrench vectors to use for computing
        # the loss. We do this by indexing into the loss vectors with the specified indices, and taking the sums.
        loss = (torch.sum(force_loss[args.predict_grf_components]) +
                torch.sum(cop_loss[args.predict_cop_components]) +
                torch.sum(moment_loss[args.predict_moment_components]) +
                torch.sum(wrench_loss[args.predict_wrench_components]))
        self.losses.append(loss)

        ############################################################################
        # Step 2: Compute report data, if we are asked to do so
        ############################################################################

        # 2.1. Initialize paper-reported values we will send to wandb, if requested
        force_err_mean: Optional[float] = None
        moment_err_mean: Optional[float] = None
        cop_err_mean: Optional[float] = None
        wrench_err_mean: Optional[float] = None
        tau_err_mean: Optional[float] = None
        com_acc_err_mean: Optional[float] = None

        if compute_report:
            with torch.no_grad():
                # 2.2. Compute the norm errors for the force, moment, CoP, and wrench vectors
                force_err_mean = RegressionLossEvaluator.get_mean_norm_error(
                    outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME],
                    labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
                ).item()
                moment_err_mean = RegressionLossEvaluator.get_mean_norm_error(
                    outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME],
                    labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]
                ).item()
                cop_err_mean = RegressionLossEvaluator.get_mean_norm_error(
                    outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor,
                    labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME] * cop_mask_tensor
                ).item()
                wrench_err_mean = RegressionLossEvaluator.get_mean_norm_error(
                    outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
                    labels[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME],
                    vec_size=6
                ).item()

                # 2.3. Manually compute the inverse dynamics torque errors frame-by-frame
                num_batches = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].shape[0]
                tau_err_mean = 0.0
                for batch in range(num_batches):
                    skel = self.dataset.skeletons[batch_subject_indices[batch]]
                    skel.setPositions(inputs[InputDataKeys.POS][batch, -1, :].cpu().numpy())
                    skel.setVelocities(inputs[InputDataKeys.VEL][batch, -1, :].cpu().numpy())
                    acc = inputs[InputDataKeys.ACC][batch, -1, :].cpu().numpy()
                    contact_bodies = self.dataset.skeletons_contact_bodies[batch_subject_indices[batch]]
                    contact_wrench_guesses = outputs[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][batch,
                                                :].cpu().numpy() * skel.getMass()
                    contact_wrench_guesses_list = [contact_wrench_guesses[i * 6:i * 6 + 6] for i in
                                                    range(len(contact_bodies))]
                    tau = skel.getInverseDynamicsFromPredictions(acc, contact_bodies, contact_wrench_guesses_list,
                                                                    np.zeros(6))
                    tau_error = tau - labels[OutputDataKeys.TAU][batch, :].cpu().numpy()
                    # Exclude root residual from error
                    tau_err_mean += np.linalg.norm(tau_error[6:])
                tau_err_mean /= (num_batches)
                self.tau_errors.append(tau_err_mean)

                # 2.4. Keep track of running sums so that we can print average values for these reportable metrics
                self.num_evaluations += 1
                self.sum_grf_forces_error += force_err_mean
                self.sum_grf_moment_error += moment_err_mean
                self.sum_grf_cop_error += cop_err_mean
                self.sum_grf_wrench_error += wrench_err_mean
                self.sum_id_tau_error += tau_err_mean

        ############################################################################
        # Step 3: Log reports to wandb and plot results, if requested
        ############################################################################

        # 3.1. If requested, log the reports to Weights and Biases
        if log_reports_to_wandb:
            self.log_to_wandb(args, force_loss, cop_loss, moment_loss, wrench_loss, loss, tau_err_mean, compute_report=compute_report)

        # 3.2. If requested, plot the results
        # if analyze:
        #     self.plot_ferror = ((force_diff) ** 2)[:, -1, :].reshape(-1, 6).detach().numpy()
        #     for i in args.predict_grf_components:
        #         plt.clf()
        #         plt.plot(self.plot_ferror[:, i])
        #         plt.savefig(os.path.join(plot_path_root,
        #                                  f"{os.path.basename(self.dataset.subject_paths[batch_subject_indices[0]])}_{self.dataset.subjects[batch_subject_indices[0]].getTrialName(batch_trial_indices[0])}_grferror{components[i]}.png"))
        return loss

    def log_to_wandb(self, args: argparse.Namespace, force_loss: torch.Tensor, cop_loss: torch.Tensor, moment_loss: torch.Tensor, wrench_loss: torch.Tensor, loss: torch.Tensor, tau_err_mean, compute_report: bool = False):
        components = {0: "left-x", 1: "left-y", 2: "left-z", 3: "right-x", 4: "right-y", 5: "right-z"}
        report: Dict[str, float] = {
            **{f'{self.split}/force_rmse/{components[i]}': force_loss[i].item()**0.5 for i in args.predict_grf_components},
            **{f'{self.split}/cop_rmse/{components[i]}': cop_loss[i].item()**0.5 for i in args.predict_cop_components},
            **{f'{self.split}/moment_rmse/{components[i]}': moment_loss[i].item()**0.5 for i in args.predict_moment_components},
            f'{self.split}/wrench_loss': torch.sum(wrench_loss).item(),
            f'{self.split}/loss': loss.item()
        }
        if compute_report:
            report[f'{self.split}/Force Avg Err (N per kg)'] = force_loss[args.predict_grf_components].mean().item()**0.5
            # report['Force Avg Err (%)'] = force_percentage_diff.mean().item()
            report[f'{self.split}/CoP Avg Err (m)'] = cop_loss[args.predict_cop_components].mean().item()**0.5
            report[f'{self.split}/Moment Avg Err (Nm per kg)'] = moment_loss[args.predict_moment_components].mean().item()**0.5
            report[f'{self.split}/Wrench Force Avg Err (N per kg)'] = (wrench_loss[3:6].mean().item() + wrench_loss[9:12].mean().item()) / 2
            report[f'{self.split}/Wrench Moment Avg Err (Nm per kg)'] = (wrench_loss[:3].mean().item() + wrench_loss[6:9].mean().item()) / 2
            report[f'{self.split}/Non-root Joint Torques (Inverse Dynamics) Avg Err (Nm per kg)'] = tau_err_mean
        #print(report)
        wandb.log(report)

    def print_report(self, args: argparse.Namespace, reset: bool = True, log_to_wandb: bool = False,
                     compute_report: bool = False):
        
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
            self.log_to_wandb(args, aggregate_force_loss, aggregate_cop_loss, aggregate_moment_loss,
                              aggregate_wrench_loss, aggregate_loss, aggregate_tau_error, compute_report=compute_report)

        if self.num_evaluations > 0:
            print(f'\tForce Avg Err: {self.sum_grf_forces_error / self.num_evaluations} N / kg')
            print(f'\tCoP Avg Err: {self.sum_grf_cop_error / self.num_evaluations} m')
            print(f'\tMoment Avg Err: {self.sum_grf_moment_error / self.num_evaluations} Nm / kg')
            print(f'\tWrench Avg Err: {self.sum_grf_wrench_error / self.num_evaluations} N+Nm / kg')
            print(f'\tNon-root Joint Torques (Inverse Dynamics) Avg Err: {self.sum_id_tau_error / self.num_evaluations} Nm')

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
            self.sum_grf_forces_error = 0.0
            self.sum_grf_forces_percent_error = 0.0
            self.sum_grf_cop_error = 0.0
            self.sum_grf_moment_error = 0.0
            self.sum_grf_wrench_error = 0.0
            self.sum_id_tau_error = 0.0
