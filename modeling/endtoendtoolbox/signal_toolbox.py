import torch

def rfft(input):
    fft = torch.rfft(input, 2, onesided=False)  
    fft = fft.squeeze(1)
    fft = fft.permute(0, 3, 1, 2)
    return fft  

def fft(input):
    
    
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=False) 

    
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=False) 
    
    input = input.permute(0, 3, 1, 2)
    return input