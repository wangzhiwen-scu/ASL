import torch

def torch_binarize(k_data, desired_sparsity):
    batch, channel, x_h, x_w = k_data.shape
    tk = int(desired_sparsity*x_h*x_w)+1
    k_data = k_data.reshape(batch, channel, x_h*x_w, 1)
    values, indices = torch.topk(k_data, tk, dim=2)
    k_data_binary =  (k_data >= torch.min(values))
    k_data_binary = k_data_binary.reshape(batch, channel, x_h, x_w).float()
    return k_data_binary