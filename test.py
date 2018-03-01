
import torch
import shift_reduce

def test_encoder():
    enc = shift_reduce.Encoder()
    buffer = torch.autograd.Variable(torch.randn(1, 10, enc.enc_size), requires_grad=False)
    return enc(buffer)

def test_decoder():
    dec = shift_reduce.Decoder()
    encoded = torch.autograd.Variable(torch.randn(1, dec.enc_size), requires_grad=False)
    return dec(encoded, buffer_length=10)