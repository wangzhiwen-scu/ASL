import torch
from torch import nn
import sys

sys.path.append('.') 
from modeling.endtoendtoolbox.sampler2d import Sampler2D, KspaceLineConstrainedSampler, LightSampler1D
from layers.mask_layer import BatchRescaleProbMap, BatchThresholdRandomMaskSigmoidV1
from modeling.endtoendtoolbox.signal_toolbox import rfft, fft, ifft
from modeling.lighting_unet import LUNet, get_parameter_number

from modeling.dualdomain import DualDoRecNet
from layers.mask_layer import ProbMask

class Sampler(nn.Module):
    def __init__(self, shape, line_constrained, mini_batch):
        super().__init__()
        
        if line_constrained:
            
            
            self.mask_net = LightSampler1D()
        else:
            self.mask_net = Sampler2D()
        
        self.rescale = BatchRescaleProbMap
        self.binarize = BatchThresholdRandomMaskSigmoidV1.apply
        self.shape = shape
        self.mini_batch = mini_batch

        self.line_constrained = line_constrained

    def forward(self, full_kspace, observed_kspace, old_mask, budget):
        sparsity = budget / (self.shape[0] * self.shape[1]) if not self.line_constrained else (budget / self.shape[0]) 
        temp = torch.cat([observed_kspace, full_kspace*old_mask], dim=1)

        
        mask = self.mask_net(temp, old_mask)
        
        binary_mask = self.binarize(mask, sparsity)
        
        binary_mask = old_mask + binary_mask

        
        binary_mask = torch.clamp(binary_mask, min=0, max=1)
        masked_kspace = binary_mask * full_kspace

        return masked_kspace, binary_mask

class MixedSampler(nn.Module):
    def __init__(self, shape, line_constrained, mini_batch, desired_sparsity):
        super().__init__()
        pmask_slope=10
        
        if line_constrained:
            
            self.mask_net = KspaceLineConstrainedSampler(in_chans=shape[0], out_chans=shape[0])
            self.learndmask = None
        else:
            self.mask_net = Sampler2D()
            self.layer_probmask = ProbMask(shape[0], shape[1], slope=pmask_slope)  
            
        
        self.rescale = BatchRescaleProbMap
        self.binarize = BatchThresholdRandomMaskSigmoidV1.apply
        self.shape = shape
        self.mini_batch = mini_batch

        self.line_constrained = line_constrained

    def forward(self, full_kspace, observed_kspace, old_mask, budget):
        sparsity = budget / (self.shape[0] * self.shape[1]) if not self.line_constrained else (budget / self.shape[0])  
        temp = torch.cat([observed_kspace, full_kspace*old_mask], dim=1)

        
        mask = self.mask_net(temp, old_mask)
        fake_input = torch.norm(full_kspace, dim=1, keepdim=True)
        loupe_prob_mask1 = self.layer_probmask(fake_input)  
        mask = mask + loupe_prob_mask1
        rescaled_mask = self.rescale(mask, sparsity)

        binary_mask = self.binarize(rescaled_mask)
        binary_mask = old_mask + binary_mask

        
        binary_mask = torch.clamp(binary_mask, min=0, max=1)
        masked_kspace = binary_mask * full_kspace

        return masked_kspace, binary_mask

class SequentialUnet(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, reconstructor_name=None):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)
        
        
        
        
        
        
        self.reconstructor = LUNet(1,1)
        
        
        
            
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained

        
        if not line_constrained:
            self.sparsity = sparsity - (preselect_num*preselect_num) / (shape[0]*shape[1]) if preselect else sparsity 
        else:
            self.sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity

        if line_constrained:
            self.budget = self.sparsity * shape[0] 
        else:
            self.budget = self.sparsity * shape[0] * shape[1]
        self.preselect_num_one_side = preselect_num // 2

    def step_forward(self, full_kspace, pred_kspace, old_mask, step): 
        
        budget = self.budget / self.num_step  

        
        masked_kspace, new_mask = self.sampler(full_kspace, pred_kspace, old_mask, budget)
        zero_filled_recon = ifft(masked_kspace) 
        zero_recon = torch.norm(zero_filled_recon, dim=1, keepdim=True)  
        recon= self.reconstructor(zero_recon)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_recon}
        return pred_dict

    def _init_mask(self, x):
        """
            Take the center 4*4 region (or 2*4 in the conjugate symmetry case)
            Up-Left, Up-right, Down-Left, Down-right + FFTShift to center
            data: NHWC (INPUT)
            a: N1HW (OUTPUT)
        """
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                """
                    In the line constrained case, we pre-select the center lines.
                """
                """if self.bidirection:
                    
                    a[:, :, :, :self.preselect_num_one_side] = 1
                    a[:, :, :self.preselect_num_one_side, :] = 1
                else:
                """
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
        """
            data: NHWC (input image in kspace)
            mask: NCHW
            kspace: NHW2

        """
        u_k0 =  f_k * initial_mask
        init_img = ifft(u_k0)
        init_img = torch.norm(init_img, dim=1, keepdim=True)
        recon = self.reconstructor(init_img)
        pred_kspace = rfft(recon)
        return pred_kspace

    def forward(self, raw_img):

        full_kspace = rfft(raw_img)

        old_mask = self._init_mask(full_kspace) 
        pred_kspace = self._init_kspace(full_kspace, old_mask)  
        
        masks = {}


        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            
            new_img, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            
            old_mask = new_mask
            pred_kspace = rfft(new_img)

            
            
            
            
            masks[i] = old_mask

        pred_dict = {'output': new_img, 'mask': new_mask, 'zero_recon': zero_recon, 'masks':masks}
        return pred_dict


class SequentialMDRecNet(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, reconstructor_name=None):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)
        self.reconstructor = DualDoRecNet(desired_sparsity=int(sparsity*100))
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained

        
        if not line_constrained:
            self.sparsity = sparsity - (preselect_num*preselect_num) / (shape[0]*shape[1]) if preselect else sparsity 
        else:
            self.sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity

        if line_constrained:
            self.budget = self.sparsity * shape[0] 
        else:
            self.budget = self.sparsity * shape[0] * shape[1]
        self.preselect_num_one_side = preselect_num // 2

    def step_forward(self, full_kspace, pred_kspace, old_mask, step): 
        
        budget = self.budget / self.num_step  

        
        masked_kspace, new_mask = self.sampler(full_kspace, pred_kspace, old_mask, budget)
        zero_filled_recon = ifft(masked_kspace) 
        
        recon= self.reconstructor(masked_kspace, zero_filled_recon, new_mask)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_filled_recon}
        return pred_dict

    def _init_mask(self, x):
        """
            Take the center 4*4 region (or 2*4 in the conjugate symmetry case)
            Up-Left, Up-right, Down-Left, Down-right + FFTShift to center
            data: NHWC (INPUT)
            a: N1HW (OUTPUT)
        """
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                """
                    In the line constrained case, we pre-select the center lines.
                """
                """if self.bidirection:
                    
                    a[:, :, :, :self.preselect_num_one_side] = 1
                    a[:, :, :self.preselect_num_one_side, :] = 1
                else:
                """
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
        """
            data: NHWC (input image in kspace)
            mask: NCHW
            kspace: NHW2

        """
        u_k0 =  f_k * initial_mask
        init_img = ifft(u_k0)
        
        recon = self.reconstructor(u_k0, init_img, initial_mask)
        pred_kspace = fft(recon)
        return pred_kspace

    def forward(self, raw_img):

        full_kspace = rfft(raw_img)

        old_mask = self._init_mask(full_kspace) 
        pred_kspace = self._init_kspace(full_kspace, old_mask)  

        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            
            new_img, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            
            old_mask = new_mask
            pred_kspace = fft(new_img)

        pred_dict = {'output': new_img, 'mask': new_mask, 'zero_recon': zero_recon}
        return pred_dict

if __name__ == '__main__':
    
    f_img = torch.randn(4,2,240,240)
    seqnet = SequentialUnet(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=0.1, preselect_num=2, mini_batch=f_img.shape[0])
    output = seqnet(f_img)
    print(get_parameter_number(seqnet))  
                                         
    x=1