import torch

# Create two 2D tensors
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([7, 8, 9])

# Stack tensors vertically (row-wise)
stacked_tensor = torch.row_stack([tensor1, tensor2])

# print(stacked_tensor)
# print(stacked_tensor.tolist())
# print(tuple(stacked_tensor.tolist()))

dict1 = {'a' : torch.tensor([[1, 2, 3],[7, 8, 9]]),
         'b' : torch.tensor([[4, 5, 6],[10, 11, 12]])}
dict2 = {'a' : torch.tensor([[1, 2, 3],[7, 8, 9]]),
         'b' : torch.tensor([[4, 5, 6],[10, 11, 12]])}

def stacked_tensor_to_tuple(t):
    return tuple(tuple(l) for l in t.tolist())

def convert_dict_to_hashable(d):
    return frozenset((k, stacked_tensor_to_tuple(v)) for k, v in d.items())

all_inputs = []
gathered = [convert_dict_to_hashable(d) for d in [dict1, dict2]]
all_inputs.extend(gathered)
print(f'all inputs: {all_inputs}')
print(f'set of inputs: {set(all_inputs)}')
assert len(all_inputs) == len(set(all_inputs)), f'Duplicate inputs detected!'