
import torch

class ResLayer(torch.nn.Module):
    def __init__(self, input_size, activation=None):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, input_size)
        self.activation = activation or torch.nn.ReLU()
    def forward(self, x):
        return self.linear(self.activation(x)) + x

# Acts like identity function on forward pass (ignores score),
# then on backward pass, behaves as though score was used to weight x
# (gradient matches x -> positive score gradient, gradient mismatches x -> negative score gradient)
class StraightThrough(torch.autograd.Function):
    def forward(self, score, x):
        self.save_for_backward(score, x)
        return x
    def backward(self, grad_out):
        score, x = self.saved_tensors
        return torch.dot(grad_out, x), grad_out

def straight_through(score, x):
    return StraightThrough()(score, x)

def decide(logits, force=None):
    scores = torch.nn.functional.softmax(logits, dim=1)
    if force is not None:
        return force, scores[0, force]
    else:
        idx = torch.multinomial(scores, 1, replacement=True).data[0, 0]
        return idx, scores[0, idx]

def distribute(module, *inputs):  # pylint: disable=arguments-differ
    reshaped_inputs = []
    for input_tensor in inputs:
        input_size = input_tensor.size()
        if len(input_size) <= 2:
            raise RuntimeError("No dimension to distribute: " + str(input_size))

        # Squash batch_size and time_steps into a single axis; result has shape
        # (batch_size * time_steps, input_size).
        squashed_shape = [-1] + [x for x in input_size[2:]]
        reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

    reshaped_outputs = module(*reshaped_inputs)

    # Now get the output back into the right shape.
    # (batch_size, time_steps, [hidden_size])
    new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
    outputs = reshaped_outputs.contiguous().view(*new_shape)

    return outputs