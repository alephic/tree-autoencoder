
import torch
import lstm_shift_reduce

def test_encoder():
    enc = lstm_shift_reduce.LSTMShiftReduceEncoder()
    buffer = torch.autograd.Variable(torch.randn(1, 10, enc.enc_size), requires_grad=False)
    return enc(buffer)

def test_decoder():
    dec = lstm_shift_reduce.LSTMShiftReduceDecoder()
    encoded = torch.autograd.Variable(torch.randn(1, dec.enc_size), requires_grad=False)
    return dec(encoded, buffer_length=10)