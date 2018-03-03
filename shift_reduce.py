
import torch
from torch.autograd import Variable
from nn_util import ResLayer, straight_through, decide
from tree_util import *

class Encoder(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.enc_size = config.get('enc_size', 256)
        lstm_size = config.get('lstm_size', 256)
        lstm_layers = config.get('lstm_layers', 2)
        self.lstm = torch.nn.LSTM(
            self.enc_size,
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
                2*self.enc_size,
                self.enc_size
            ),
            ResLayer(self.enc_size)
        )

    def forward(self, buffer, fixed_actions=None):
        # buffer : (batch_size=1, buffer_size, enc_size)
        buffer_index = 0
        buffer_size = buffer.size(1)
        stack = []
        # unsqueeze(1) -> (num_layers, batch_size=1, lstm_size)
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
                    # stack[0] : (batch_size=1, enc_size)
                    # torch.stack(action_logit_record, 1) : (batch_size=1, len(action_record), 2)
                    return stack[0], action_record, torch.stack(action_logit_record, 1) # return final encoding
            if force is None and fixed_actions is not None:
                force = fixed_actions[len(action_record)]
            # state_stack[-1][0] = h : (num_layers, batch_size=1, lstm_size)
            # state_stack[-1][0][-1] : (batch_size=1, lstm_size)
            act_logits = self.act_scorer(state_stack[-1][0][-1]) # (batch_size=1, 2)
            act_idx, act_score = decide(act_logits, force=force)
            action_record.append(act_idx)
            action_logit_record.append(act_logits)
            if act_idx == 0: # shift
                shifted = buffer[:, buffer_index] # (batch_size=1, enc_size)
                buffer_index += 1
                # shifted.unsqueeze(1) : (batch_size=1, seq_length=1, enc_size)
                # state_stack[-1] = h, c : (num_layers, batch_size=1, lstm_size)
                _, new_state = self.lstm(shifted.unsqueeze(1), state_stack[-1]) # new_state = h, c : (num_layers, batch_size=1, lstm_size)
                stack.append(straight_through(act_score, shifted))
                state_stack.append((
                    straight_through(act_score, new_state[0]),
                    straight_through(act_score, new_state[1])
                ))
            elif act_idx == 1: # reduce
                prev_enc_r = stack.pop()
                prev_state_r = state_stack.pop()
                prev_enc_l = stack.pop()
                prev_state_l = state_stack.pop()
                reduced = self.reduce(
                    torch.cat((
                        prev_enc_r, # (batch_size=1, enc_size)
                        prev_enc_l
                    ), 1) # (batch_size=1, 2*enc_size)
                ) # (batch_size=1, enc_size)
                # reduced.unsqueeze(1) : (batch_size=1, seq_length=1, enc_size)
                _, new_state = self.lstm(reduced.unsqueeze(1), state_stack[-1]) # new_state = h, c : (num_layers, batch_size=1, lstm_size)
                stack.append(straight_through(act_score, reduced))
                state_stack.append((
                    straight_through(act_score, new_state[0]),
                    straight_through(act_score, new_state[1])
                ))

class Decoder(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.enc_size = config.get('enc_size', 256)
        self.act_scorer = torch.nn.Sequential(
            ResLayer(self.enc_size),
            torch.nn.Linear(
                self.enc_size,
                2
            )
        )
        self.unreduce_l = torch.nn.Sequential(
            ResLayer(self.enc_size),
            ResLayer(self.enc_size)
        )
        self.unreduce_r = torch.nn.Sequential(
            ResLayer(self.enc_size),
            ResLayer(self.enc_size)
        )

    def forward(self, input_encoding, buffer_length=None, fixed_actions=None):
        # input_encoding : (batch_size=1, enc_size)
        # Decoder fixed actions are time-reversed encoder actions
        buffer_slices = []
        stack = [input_encoding]
        action_record = []
        action_logit_record = []
        while True:
            force = None
            if buffer_length is not None:
                if len(stack) == 1 and len(buffer_slices) + 1 < buffer_length:
                    # unshifting the last remaning token would result in too small of a buffer
                    force = 1 # force unreduce
                elif len(stack) + len(buffer_slices) == buffer_length:
                    # unreducing any more would generate too large of a buffer
                    force = 0 # force unshift
            if force is None and fixed_actions is not None:
                force = fixed_actions[len(action_record)]
            act_logits = self.act_scorer(stack[-1]) # (batch_size=1, 2)
            act_idx, act_score = decide(act_logits, force=force)
            action_record.append(act_idx)
            action_logit_record.append(act_logits)
            if act_idx == 0: # unshift
                unshifted = stack.pop() # (batch_size=1, enc_size)
                buffer_slices.append(straight_through(act_score, unshifted))
                if len(stack) == 0:
                    # torch.stack(buffer_slices, 1) -> (batch_size=1, len(buffer_slices), enc_size)
                    # torch.stack(action_logit_record, 1) -> (batch_size=1, len(action_record), 2)
                    return torch.stack(buffer_slices, 1), action_record, torch.stack(action_logit_record, 1)
            elif act_idx == 1: # unreduce
                prev_enc = stack.pop() # (batch_size=1, enc_size)
                unreduced_l = self.unreduce_l(prev_enc) # (batch_size=1, enc_size)
                unreduced_r = self.unreduce_r(prev_enc) # (batch_size=1, enc_size)
                stack.append(straight_through(act_score, unreduced_r))
                stack.append(straight_through(act_score, unreduced_l))
