
import torch

class ResLayer(torch.nn.Module):
    def __init__(self, input_size, activation=None):
        self.linear = torch.nn.Linear(input_size, input_size)
        self.activation = activation or torch.nn.ReLU()
    def forward(self, x):
        return self.linear(self.activation(x)) + x

class StraightThrough(torch.autograd.Function):
    def forward(self, score, x):
        self.save_for_backward(score, x)
        return x
    def backward(self, grad_out):
        score, x = self.saved_tensors
        return torch.dot(grad_out, x), grad_out

def decide(logits, force=None):
    scores = F.softmax(logits, dim=1)
    if force is not None:
        return force, scores[force]
    else:
        idx = torch.multinomial(scores, 1, with_replacement=True).data[0, 0]
        return idx, scores[idx]