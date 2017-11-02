
import torch
from torch.autograd import Variable
from nn_util import ResLayer, straight_through, decide, distribute

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
        self.act_scorer = torch.nn.Linear(
            lstm_size,
            2
        )
        self.reduce = torch.nn.Sequential(
            torch.nn.Linear(
                2*(enc_size + lstm_size),
                enc_size
            ),
            ResLayer(enc_size)
        )

    def forward(self, input_indices, fixed_actions=None):
        buffer = self.embed(input_indices) # (batch_size=1, buffer_size, enc_size)
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
                        prev_state_r[0][-1], # (batch_size=1, lstm_size)
                        prev_enc_l
                        prev_state_l[0][-1],
                    ), 1) # (batch_size=1, 2*(enc_size + lstm-size))
                ) # (batch_size=1, enc_size)
                # reduced.unsqueeze(1) : (batch_size=1, seq_length=1, enc_size)
                _, new_state = self.lstm(reduced.unsqueeze(1), state_stack[-1]) # new_state = h, c : (num_layers, batch_size=1, lstm_size)
                stack.append(straight_through(act_score, reduced))
                state_stack.append((
                    straight_through(act_score, new_state[0]),
                    straight_through(act_score, new_state[1])
                ))

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

    def forward(self, input_encoding, buffer_length=None, fixed_actions=None):
        # input_encoding : (batch_size=1, enc_size)
        buffer_slices = []
        stack = [input_encoding]
        # unsqueeze input_encoding -> (batch_size=1, seq_length=1, enc_size)
        # unsqueeze h0, c0 -> (num_layers, batch_size=1, lstm_size)
        _, init_state = self.lstm(input_encoding.unsqueeze(1), (self.h0.unsqueeze(1), self.c0.unsqueeze(1)))
        state_stack = [init_state]
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
            # state_stack[-1][0] = h : (num_layers, batch_size=1, lstm_size)
            # state_stack[-1][0][-1] : (batch_size=1, lstm_size)
            act_logits = self.act_scorer(state_stack[-1][0][-1]) # (batch_size=1, 2)
            act_idx, act_score = decide(act_logits, force=force)
            action_record.append(act_idx)
            action_logit_record.append(act_logits)
            if act_idx == 0: # unshift
                unshifted = stack.pop() # (batch_size=1, enc_size)
                state_stack.pop() # h, c : (num_layers, batch_size=1, lstm_size)
                buffer_slices.append(straight_through(act_score, unshifted))
                if len(stack) == 0:
                    # torch.stack(buffer_slices, 1) -> (batch_size=1, buffer_size, enc_size)
                    # torch.stack(action_logit_record, 1) -> (batch_size=1, len(action_record), 2)
                    return torch.stack(buffer_slices, 1), action_record, torch.stack(action_logit_record, 1)
            elif act_idx == 1: # unreduce
                prev_enc = stack.pop() # (batch_size=1, enc_size)
                prev_h, prev_c = state_stack.pop() # (num_layers, batch_size=1, lstm_size)
                unreduce_in = torch.cat((
                    prev_enc,
                    prev_h[-1]
                ), 1) # (batch_size=1, enc_size + lstm_size)
                unreduced_l = self.unreduce_l(unreduce_in) # (batch_size=1, enc_size)
                unreduced_r = self.unreduce_r(unreduce_in) # (batch_size=1, enc_size)
                # torch.cat((unreduced_l, unreduced_r), 0) -> (batch_size=2, enc_size)
                # prev_{h, c}.expand(-1, 2, -1) -> (num_layers, batch_size=2, lstm_size)
                _, new_states = self.lstm(torch.cat((unreduced_l, unreduced_r), 0), (prev_h.expand(-1, 2, -1), prev_c.expand(-1, 2, -1)))
                new_state_l, new_state_r = torch.chunk(new_states, 2, 1) # (num_layers, batch_size=1, lstm_size)
                stack.append(straight_through(act_score, unreduced_l))
                stack.append(straight_through(act_score, unreduced_r))
                state_stack.append((
                    straight_through(act_score, new_state_l[0]),
                    straight_through(act_score, new_state_l[1])
                ))
                state_stack.append((
                    straight_through(act_score, new_state_r[0]),
                    straight_through(act_score, new_state_r[1])
                ))
