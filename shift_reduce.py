
import torch
from torch.autograd import Variable
from nn_util import ResLayer, StraightThrough, decide

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
            enc_size + 2, # 2d one-hot encoding of action
            lstm_size,
            lstm_layers,
            batch_first=True
        )
        self.h0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.c0 = torch.nn.Parameter(torch.Tensor(lstm_layers, lstm_size).zero_())
        self.register_buffer('act_shift', Variable(torch.Tensor([1.0, 0.0]), requires_grad=False))
        self.register_buffer('act_reduce', Variable(torch.Tensor([0.0, 1.0]), requires_grad=False))
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
    
    def forward(self, input_indices, fixed_actions=None):
        buffer = self.embed(input_indices) # (batch_size=1, buffer_size, enc_size)
        buffer_index = 0
        buffer_size = buffer.size(1)
        stack = [] # [((h, c), enc)]
        state_stack = [(self.h0.unsqueeze(1), self.c0.unsqueeze(1))]
        action_record = []
        action_logit_record = []
        while True:
            force = None
            if len(stack) < 2: # not enough stack to reduce
                force = 0 # force shift
            if buffer_index == buffer_size: # not enough buffer to shift
                if force is None: # there's enough stack to reduce
                    force = 1 # force reduce
                else: # can't shift or reduce, we're done
                    return stack[0][1], action_record, torch.stack(action_logit_record, 1) # return final encoding
            act_logits = self.act_scorer(state_stack[-1][0][-1])
            act_idx, act_score = decide(act_logits, force=force)
            action_record.append(act_idx)
            action_logit_record.append(act_logits)
            if act_idx == 0: # shift
                shifted = buffer[:, buffer_index]
                buffer_index += 1
                _, new_state = self.lstm(
                    StraightThrough()(
                        act_score,
                        torch.cat((
                            shifted,
                            self.act_shift
                        ), 1)
                    ).unsqueeze(1),
                    state_stack[-1]
                )
                stack.append(StraightThrough()(act_score, shifted))
                state_stack.append(new_state)
            elif act_idx == 1: # reduce
                prev_state_r = state_stack.pop()
                prev_enc_r = stack.pop()
                prev_state_l = state_stack.pop()
                prev_enc_l = stack.pop()
                reduced = self.reduce(
                    torch.cat((
                        prev_state_r[-1],
                        prev_enc_r,
                        prev_state_l[-1],
                        prev_enc_l
                    ), 1)
                )
                _, new_state = self.lstm(
                    StraightThrough()(
                        act_score,
                        torch.cat((
                            reduced,
                            self.act_reduce
                        ), 1)
                    ).unsqueeze(1),
                    state_stack[-1]
                )
                stack.append(StraightThrough()(act_score, reduced))
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
        self.register_buffer('act_unshift', Variable(torch.Tensor([1.0, 0.0]), requires_grad=False))
        self.register_buffer('act_unreduce', Variable(torch.Tensor([0.0, 1.0]), requires_grad=False))
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
    def forward(self, input_encoding, fixed_actions=None, buffer_length=None):
        pass