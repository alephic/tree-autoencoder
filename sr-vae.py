
import torch
from torch.autograd import Variable

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

class ShiftReduceEncoder(torch.nn.Module):
    def __init__(self, **config):
        self.config = config
        enc_size = config.get('enc_size', 256)
        vocab_size = config['vocab_size']
        self.embed = torch.nn.Embedding(
            vocab_size,
            enc_size
        )
        lstm_size = config.get('lstm_size', 256)
        lstm_layers = config.get('lstm_layers', 2)
        self.lstm = torch.nn.LSTM(
            enc_size,
            lstm_size,
            lstm_layers,
            batch_first=True
        )
        self.h0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.c0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.act_scorer = torch.nn.Linear(
            lstm_size,
            2
        )
        self.reduce = torch.nn.Sequential(
            torch.nn.Linear(
                2*(enc_size + lstm_layers*lstm_size),
                enc_size
            ),
            ResLayer(enc_size)
        )
    
    def forward(self, input_indices):
        buffer = self.embed(input_indices) # (batch_size=1, buffer_size, enc_size)
        buffer_index = 0
        buffer_size = buffer.size(1)
        stack = [] # [((h, c), enc)]
        state_stack = [(self.h0.unsqueeze(1), self.c0.unsqueeze(1))]
        action_record = []
        while True:
            force = None
            if len(stack) < 2: # not enough stack to reduce
                force = 0 # force shift
            if buffer_index == buffer_size: # not enough buffer to shift
                if force is None: # there's enough stack to reduce
                    force = 1 # force reduce
                else: # can't shift or reduce, we're done
                    return stack[0][1], action_record # return final encoding
            act_idx, act_score = decide(self.act_scorer(state_stack[-1][0][-1]), force=force)
            action_record.append(act_idx)
            if act_idx == 0: # shift
                shifted = StraightThrough()(act_score, buffer[:, buffer_index])
                buffer_index += 1
                _, new_state = self.lstm(shifted.unsqueeze(1), state_stack[-1])
                stack.append(shifted)
                state_stack.append(new_state)
            elif act_idx == 1: # reduce
                prev_state_r = state_stack.pop()
                prev_enc_r = stack.pop()
                prev_state_l = state_stack.pop()
                prev_enc_l = stack.pop()
                reduced = StraightThrough()(
                    act_score,
                    self.reduce(torch.cat(
                        (prev_state_r[-1], prev_enc_r, prev_state_l[-1], prev_enc_l),
                        1
                    ))
                )
                _, new_state = self.lstm(reduced.unsqueeze(1), state_stack[-1])
                stack.append(reduced)
                state_stack.append(new_state)

class ShiftReduceDecoder(torch.nn.Module):
    def __init__(self, **config):
        self.config = config
        enc_size = config.get('enc_size', 256)
        vocab_size = config['vocab_size']
        lstm_size = config.get('lstm_size', 256)
        lstm_layers = config.get('lstm_layers', 2)
        self.lstm = torch.nn.LSTM(
            enc_size,
            lstm_size,
            lstm_layers,
            batch_first=True
        )
        self.h0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.c0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.act_scorer = torch.nn.Linear(
            lstm_size,
            2
        )
        self.unreduce_l = torch.nn.Sequential(
            torch.nn.Linear(
                enc_size + lstm_size,
                enc_size
            ),
            ResLayer(enc_size)
        )
        self.unreduce_r = torch.nn.Sequential(
            torch.nn.Linear(
                enc_size + lstm_size,
                enc_size
            ),
            ResLayer(enc_size)
        )