import sys
import torch
from torch import nn

sys.path.append('.') 
from modeling.recon_net import ReconUnetforComb, CSMRIUnet
from modeling.seg_net import SimpleUnet

from utils.train_utils import get_weights


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Comb_Net(nn.Module):
    def __init__(self, cartesian, nclasses, inputs_size, desired_sparsity, isfinetune, isfixseg, ckpt, isDC, args, sample_slope=10):
        super(Comb_Net, self).__init__()
        self.recon_net =  ReconUnetforComb(cartesian=cartesian, inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, isDC=isDC, isfinetune=isfinetune, ckpt=ckpt)

        self.seg_net = SimpleUnet(1, nclasses)
        if isfixseg:

            seg_path ="./model_zoo/pretrained_seg/{}_{}seg.pth".format(args.dataset_name, nclasses)
            get_weights(seg_path, self.seg_net)
            for param in self.seg_net.parameters():
                param.requires_grad = False
            print(seg_path)


    def forward(self, x):
        out_recon, uifft, complex_abs, mask, fft, undersample = self.recon_net(x)
        out_seg = self.seg_net(out_recon)
        return out_recon, out_seg, uifft, complex_abs, mask, fft, undersample



class CSMTLNet(nn.Module):
    def __init__(self, args, nclasses, inputs_size, test=False):
        super(CSMTLNet, self).__init__()
        self.recon_net = CSMRIUnet(args.rate, args.mask, inputs_size)
        self.seg_net = SimpleUnet(1, nclasses)

        
        if test:
            pass
        else:
            seg_path ="./model_zoo/pretrained_seg/{}_{}seg.pth".format(args.dataset_name, nclasses)
            get_weights(seg_path, self.seg_net)
    
    def forward(self, x):
        out_recon, uifft, complex_abs, mask, fft, undersample = self.recon_net(x)
        out_seg = self.seg_net(out_recon)
        return out_recon, out_seg, uifft, complex_abs, mask, fft, undersample

