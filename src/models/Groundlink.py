import torch
import torch.nn as nn
from typing import List
from data.AddBiomechanicsDataset import InputDataKeys, OutputDataKeys
import logging
from functorch import combine_state_for_ensemble, vmap

class Transpose(nn.Module):
	def __init__(self, dim1, dim2):
		super().__init__()
		self._dim1, self._dim2 = dim1, dim2

	def extra_repr(self):
		return "{}, {}".format(self._dim1, self._dim2)

	def forward(self, input):
		return input.transpose(self._dim1, self._dim2)

class Groundlink(nn.Module):
	def __init__(self, num_dofs: int, num_joints: int, root_history_len:int, output_data_format: str = "all_frames", cnn_kernel=7, cnn_dropout=0.0, fc_depth=3, fc_dropout=0.2):
		super().__init__()
		self.num_dofs = num_dofs
		self.num_joints = num_joints
		self.root_history_len = root_history_len
		self.output_data_format = output_data_format
		input_size = (num_dofs * 3 + 12 + num_joints * 3 + root_history_len * 6)
		cnn_features = [input_size, 128, 128, 256, 256]
		# features_out = 3
		# features_out = 5
		features_out = 30
		# features_out = 16

		def get_layers():
			## Preprocess part
			pre_layers = [																# N x F x J x [...]
				torch.nn.Flatten(start_dim=2, end_dim=-1),								# N x F x C
				Transpose(-2, -1),														# N x C x F
			]		
			
			## Convolutional part
			conv = lambda c_in, c_out: torch.nn.Conv1d(c_in, c_out, cnn_kernel, padding=cnn_kernel//2, padding_mode="replicate")
			cnn_layers = []
			for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):				# N x C x F
				cnn_layers += [
					torch.nn.Dropout(p=cnn_dropout),									# N x Ci x F
					conv(c_in, c_out),													# N x Ci x F
					torch.nn.ELU(),														# N x Ci x F
				]
			
			## Fully connected part
			fc_layers = [Transpose(-2, -1)]												# N x F x Cn
			for _ in range(fc_depth - 1):
				fc_layers += [															# N x F x Ci
					torch.nn.Dropout(p=fc_dropout),										# N x F x Ci
					torch.nn.Linear(cnn_features[-1], cnn_features[-1]),				# N x F x Ci
					torch.nn.ELU()														# N x F x Ci
				]
			fc_layers += [																# N x F x Ci
				torch.nn.Dropout(p=fc_dropout),											# N x F x 2*Co
				torch.nn.Linear(cnn_features[-1], features_out, bias=False),						# N x F x 2*Co
				# torch.nn.Softplus(),													# N x F x 2*Co
			]
			return pre_layers, cnn_layers, fc_layers
		
		# self.per_output_nets = nn.ModuleList()
		# for _ in range(features_out):
		# 	net: List[nn.Module] = get_layers()
		# 	self.per_output_nets.append(self.initialize(nn.Sequential(*net)))
		# logging.info(f"{self.per_output_nets=}")

		# self.fmodel, self.params, self.buffers = combine_state_for_ensemble(self.per_output_nets)
		# [p.requires_grad_() for p in self.params]
		pre, cnn, fc = get_layers()
		self.pre_net = self.initialize(nn.Sequential(*pre))
		self.cnn = self.initialize(nn.Sequential(*cnn))
		self.fc = self.initialize(nn.Sequential(*fc))
		logging.info(f"{self.pre_net=}, {self.cnn=}, {self.fc=}")

	def initialize(self, net):
		GAINS = {
			torch.nn.Sigmoid:	torch.nn.init.calculate_gain("sigmoid"),
			torch.nn.ReLU:		torch.nn.init.calculate_gain("relu"),
			torch.nn.LeakyReLU:	torch.nn.init.calculate_gain("leaky_relu"),
			torch.nn.ELU:		torch.nn.init.calculate_gain("relu"),
			torch.nn.Softplus:	torch.nn.init.calculate_gain("relu"),
		}
		for layer, activation in zip(list(net)[:-1], list(net)[1:]):
			if len(list(layer.parameters())) > 0 and type(activation) in GAINS:
				if not isinstance(activation, type):
					activation = type(activation)
				if activation not in GAINS:
					raise Exception("Initialization not defined for activation '{}'.".format(type(activation)))
				if isinstance(layer, torch.nn.Linear):
					torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
					if layer.bias is not None:
						torch.nn.init.zeros_(layer.bias)
				elif isinstance(layer, torch.nn.Conv1d):
					torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
					if layer.bias is not None:
						torch.nn.init.zeros_(layer.bias)
				else:
					raise Exception("Initialization not defined for layer '{}'.".format(type(layer)))
		return net

	def forward(self, input):
		# 1. Check input shape matches our assumptions.
		assert len(input[InputDataKeys.POS].shape) == 3
		assert input[InputDataKeys.POS].shape[-1] == self.num_dofs
		assert len(input[InputDataKeys.VEL].shape) == 3
		assert input[InputDataKeys.VEL].shape[-1] == self.num_dofs
		assert len(input[InputDataKeys.ACC].shape) == 3
		assert input[InputDataKeys.ACC].shape[-1] == self.num_dofs
		assert len(input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape) == 3
		assert input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME].shape[-1] == self.num_joints * 3
		assert len(input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape) == 3
		assert input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3
		assert len(input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape) == 3
		assert input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME].shape[-1] == self.root_history_len * 3

		# 2. Concatenate inputs, and flatten them to be a single long vector for each batch entry. That means each
		# timestep in the input data gets concatenated end-to-end.
		inputs = torch.concat([
			input[InputDataKeys.POS],
			input[InputDataKeys.VEL],
			input[InputDataKeys.ACC],
			input[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME],
			input[InputDataKeys.ROOT_ANGULAR_VEL_IN_ROOT_FRAME],
			input[InputDataKeys.ROOT_LINEAR_ACC_IN_ROOT_FRAME],
			input[InputDataKeys.ROOT_ANGULAR_ACC_IN_ROOT_FRAME],
			input[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME],
			input[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME],
			input[InputDataKeys.ROOT_EULER_HISTORY_IN_ROOT_FRAME]
		], dim=-1)
		
		# 3. Run one forward model stack per output variable. Each stack is an MLP. The reason to have separate stacks
		# per output is that we are doing a regression with different scales for each output, and we don't want to
		# have to worry about scaling the outputs to be the same scale. So, we just train separate models for each
		# output variable.
		# x = torch.cat([net(inputs) for net in self.per_output_nets], dim=-1)
		# x = vmap(self.fmodel, in_dims=(0, 0, None), randomness='different')(self.params, self.buffers, inputs)
		# x = torch.swapaxes(x, 0, -1).squeeze(0)
		# x = self.per_output_nets[0](inputs)
		x = self.pre_net(inputs)
		x = self.cnn(x)
		if self.output_data_format == 'all_frames':
			x = self.fc(x)
		else:
			x = self.fc(x[:, :, -1:])
		# 4. Break the output back up into separate tensors for each output variable. Predicts just a single frame of
		# output, given the input context.
		return {
			OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME: x[:, :, 0:6],
			OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME: x[:, :, 6:12],
			OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME: x[:, :, 12:18],
			OutputDataKeys.GROUND_CONTACT_WRENCHES_IN_ROOT_FRAME: x[:, :, 18:30]
		}
