import torch

def rfft(input):
    """ 
    x   - [12, 1, 240, 240]
    fft - [12, 1, 240, 240, 2]  (squeeze)-> [12, 240, 240, 2]
    """
    # https://pytorch.org/docs/1.7.1/generated/torch.rfft.html?highlight=torch%20rfft#torch.rfft
    fft = torch.rfft(input, 2, onesided=False)  # Real-to-complex Discrete Fourier Transform. normalized =False
    fft = fft.squeeze(1)
    fft = fft.permute(0, 3, 1, 2)
    return fft

def fft(input):
    # (N, 2, W, H) -> (N, W, H, 2)
    # print(type(input))
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=False) # !

    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=False) # !

    # (N, D, W, H, 2) -> (N, 2, D, W, H)
    input = input.permute(0, 3, 1, 2)

    return input