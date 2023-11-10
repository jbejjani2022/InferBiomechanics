from typing import Dict
import torch
import torch.nn as nn
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys, AddBiomechanicsDataset
import nimblephysics as nimble
import numpy as np
from typing import List, Tuple


class AnalyticalBaseline(nn.Module):

    def __init__(self):
        super(AnalyticalBaseline, self).__init__()

    def forward(self, input: Dict[str, torch.Tensor], skels_and_contact: List[Tuple[nimble.dynamics.Skeleton, List[nimble.dynamics.BodyNode]]]) -> Dict[str, torch.Tensor]:
        # input[InputDataKeys.POS],
        # input[InputDataKeys.VEL],
        # input[InputDataKeys.ACC],
        # input[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME],
        # input[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
        # input[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME],
        # input[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME],

        num_contact_bodies = len(skels_and_contact[0][1])
        num_dofs = input[InputDataKeys.POS].shape[-1]

        with torch.no_grad():
            output_dict: Dict[str, torch.Tensor] = {
                OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0],
                                                                                  input[InputDataKeys.POS].shape[1],
                                                                                  6 * num_contact_bodies),
                OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0],
                                                                                input[InputDataKeys.POS].shape[1],
                                                                                3 * num_contact_bodies),
                OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0],
                                                                                input[InputDataKeys.POS].shape[1],
                                                                                3 * num_contact_bodies),
                OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0],
                                                                                 input[InputDataKeys.POS].shape[1],
                                                                                 3 * num_contact_bodies),
                OutputDataKeys.RESIDUAL_WRENCH_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0], input[InputDataKeys.POS].shape[1], 6),
                OutputDataKeys.CONTACT: torch.zeros(input[InputDataKeys.POS].shape[0], input[InputDataKeys.POS].shape[1], num_contact_bodies),
                OutputDataKeys.COM_ACC_IN_ROOT_FRAME: torch.zeros(input[InputDataKeys.POS].shape[0], input[InputDataKeys.POS].shape[1], 3),
                OutputDataKeys.TAU: torch.zeros(input[InputDataKeys.POS].shape[0], input[InputDataKeys.POS].shape[1], num_dofs),
            }

            num_batches = input[InputDataKeys.POS].shape[0]
            num_timesteps = input[InputDataKeys.POS].shape[1]
            for batch in range(num_batches):
                skel = skels_and_contact[batch][0]
                contact_bodies = skels_and_contact[batch][1]

                for timestep in range(num_timesteps):
                    skel.setPositions(input[InputDataKeys.POS][batch, timestep, :].cpu().numpy())
                    skel.setVelocities(input[InputDataKeys.VEL][batch, timestep, :].cpu().numpy())
                    skel.setAccelerations(input[InputDataKeys.ACC][batch, timestep, :].cpu().numpy())

                    # Compute the COM acceleration
                    com_acc = skel.getCOMLinearAcceleration() - skel.getGravity()

                    # Get each foot's height and velocity
                    contact = [0.0 for _ in contact_bodies]
                    for i, body in enumerate(contact_bodies):
                        velocity = np.linalg.norm(body.getCOMLinearVelocity())
                        height = body.getWorldTransform().translation()[1]
                        if height < 0.1: # and velocity < 0.1:
                            # Let's say the foot is in contact
                            contact[i] = 1.0

                    if sum(contact) == 0:
                        # No contact, so no ground contact forces
                        continue

                    root_body = skel.getRootBodyNode()
                    T_wr = root_body.getWorldTransform()
                    T_rw = T_wr.inverse()

                    # Compute the contact forces
                    # if int(sum(contact)) == len(contact_bodies):  # double-support, distribute forces with heuristic
                    #     InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME[batch, timestep, pelvis ix]
                    #     InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME[batch, timestep, right heel ix]
                    #     InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME[batch, timestep, left heel ix]
                    #     # get ratio in each coordinate direction for each foot
                    # else:
                    world_contact_forces = [contact[i] * (com_acc / sum(contact)) for i in range(len(contact_bodies))]

                    root_contact_forces = [T_rw.rotation() @ contact_force for contact_force in world_contact_forces]
                    contact_moments = [np.zeros(3) for _ in contact_bodies]
                    world_cops = [body.getCOM() for body in contact_bodies]
                    root_cops = [T_rw.multiply(body.getCOM()) for body in contact_bodies]

                    output_dict[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][batch, timestep, :] = torch.tensor(np.row_stack(root_contact_forces)).flatten()
                    output_dict[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][batch, timestep, :] = torch.tensor(np.row_stack(root_cops)).flatten()

                    # Compute the wrenches
                    for i in range(len(contact_bodies)):
                        moment = np.cross(world_cops[i], world_contact_forces[i])
                        world_wrench: np.ndarray = np.concatenate((moment, world_contact_forces[i]))
                        body_wrench = nimble.math.dAdInvT(T_wr.rotation(), T_wr.translation(), world_wrench)
                        output_dict[OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME][batch, timestep, i*6:i*6+6] = torch.tensor(body_wrench)

            return output_dict

