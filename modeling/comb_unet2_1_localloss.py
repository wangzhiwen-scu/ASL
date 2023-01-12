import sys

import torch
from torch import nn

sys.path.append('.')

from modeling.seg_net import SimpleUnet
from utils.train_utils import get_weights
from runner.spade.spadeModule import SemanticReconNet 


from modeling.endtoendtoolbox.samplenet_istanet import Sampler
from modeling.endtoendtoolbox.signal_toolbox import rfft, fft, ifft

from modeling.recon_net import ReconUnet
from modeling.dualdomain import DualDoRecNet


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequentialASLPreTrained(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, inputs_size, nclasses, args):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)

        self.reconstructor = ReconUnet(1,1)
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        self.seg_net = SimpleUnet(1, nclasses)

        seg_path = args.segpath
        get_weights(seg_path, self.seg_net)
        for param in self.seg_net.parameters():
            param.requires_grad = False
        print(seg_path)

        
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
        recon = self.reconstructor(zero_recon)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_recon}
        return pred_dict

    def _init_mask(self, x):

        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):

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

        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            
            out_seg = self.seg_net(first_recon)  
            
            second_rec = first_recon
            old_mask = new_mask
            pred_kspace = rfft(first_recon)

        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon, 'out_seg':out_seg, 'second_rec': second_rec}
        return pred_dict

class SequentialASL(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, inputs_size, nclasses, args):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)

        self.reconstructor = ReconUnet(1,1)
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        self.seg_net = SimpleUnet(1, nclasses)
        self.fine_recon_net = SemanticReconNet(input_nc=0, output_nc=1, out_size=inputs_size[0], n_class=nclasses)

        
        seg_path = args.segpath
        get_weights(seg_path, self.seg_net)
        for param in self.seg_net.parameters():
            param.requires_grad = False
        print(seg_path)

        
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
        recon = self.reconstructor(zero_recon)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_recon}
        return pred_dict

    def _init_mask(self, x):

        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:

                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
        u_k0 =  f_k * initial_mask
        init_img = ifft(u_k0)
        init_img = torch.norm(init_img, dim=1, keepdim=True)
        recon= self.reconstructor(init_img)
        pred_kspace = rfft(recon)
        return pred_kspace

    def forward(self, raw_img):

        full_kspace = rfft(raw_img)

        old_mask = self._init_mask(full_kspace) 
        pred_kspace = self._init_kspace(full_kspace, old_mask)  

        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            
            out_seg = self.seg_net(first_recon)  
            second_rec =self.fine_recon_net(first_recon, torch.sigmoid(out_seg))

            
            old_mask = new_mask
            pred_kspace = rfft(second_rec)
            
        out_seg = self.seg_net(second_rec)
        second_rec =self.fine_recon_net(second_rec, torch.sigmoid(out_seg))

        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon, 'out_seg':out_seg, 'second_rec': second_rec}
        return pred_dict

class SequentialASLDualDoPreTrained(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, inputs_size, nclasses, args):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)

        self.reconstructor = DualDoRecNet(desired_sparsity=int(sparsity*100))
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        self.seg_net = SimpleUnet(1, nclasses)
        seg_path = args.segpath
        get_weights(seg_path, self.seg_net)
        for param in self.seg_net.parameters():
            param.requires_grad = False
        print(seg_path)

        
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
        
        recon = self.reconstructor(masked_kspace, zero_filled_recon, new_mask)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_filled_recon}
        return pred_dict

    def _init_mask(self, x):
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
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
            
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            first_recon_abs = torch.norm(first_recon, dim=1, keepdim=True)  
            
            out_seg = self.seg_net(first_recon_abs)  
            
            second_rec = first_recon.detach()
            old_mask = new_mask
            pred_kspace = fft(first_recon)
            

        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon, 'out_seg':out_seg, 'second_rec': second_rec}
        return pred_dict

class SequentialASLDualDoStage2(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, inputs_size, nclasses, args):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)

        self.reconstructor = DualDoRecNet(desired_sparsity=int(sparsity*100))
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        self.seg_net = SimpleUnet(1, nclasses)
        self.args = args
        if args.dataset_name == 'ACDC':
            n_channles = 4
        else:
            n_channles = 1  
        self.fine_recon_net = SemanticReconNet(input_nc=0, output_nc=2, out_size=inputs_size[0], n_class=nclasses+n_channles) 
        seg_path = args.segpath
        get_weights(seg_path, self.seg_net)
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
        
        recon = self.reconstructor(masked_kspace, zero_filled_recon, new_mask)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_filled_recon}
        return pred_dict

    def _init_mask(self, x):
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
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
            
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            first_recon_abs = torch.norm(first_recon, dim=1, keepdim=True)  

            out_seg = self.seg_net(first_recon_abs)  

            if self.args.dataset_name == 'ACDC':
                sec_input = torch.cat([first_recon_abs, first_recon, zero_recon], dim=1)  
            else:
                sec_input = first_recon
            second_rec =self.fine_recon_net(sec_input, torch.sigmoid(out_seg))

            old_mask = new_mask
            
            pred_kspace = fft(second_rec)
        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon, 'out_seg':out_seg, 'second_rec': second_rec}
        return pred_dict

class SequentialASLDualDo(nn.Module):
    def __init__(self, num_step, shape, preselect, line_constrained, sparsity, preselect_num, mini_batch, inputs_size, nclasses, args):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)

        self.reconstructor = DualDoRecNet(desired_sparsity=int(sparsity*100))
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        self.seg_net = SimpleUnet(1, nclasses)
        self.args = args

        if args.dataset_name == 'ACDC':
          if args.rate == 0.05:
              n_channles = 3 
          elif args.rate == 0.1 or self.args.maskType == '1D':
              n_channles = 4 
          else:
            n_channles = 1
        else:
            n_channles = 1  

        self.fine_recon_net = SemanticReconNet(input_nc=0, output_nc=2, out_size=inputs_size[0], n_class=nclasses+n_channles) 

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
        
        recon = self.reconstructor(masked_kspace, zero_filled_recon, new_mask)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_filled_recon}
        return pred_dict

    def _init_mask(self, x):
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
        u_k0 =  f_k * initial_mask
        init_img = ifft(u_k0)
        
        recon = self.reconstructor(u_k0, init_img, initial_mask)
        pred_kspace = fft(recon)
        return pred_kspace

    def forward(self, raw_img):

        full_kspace = rfft(raw_img)

        old_mask = self._init_mask(full_kspace) 
        pred_kspace = self._init_kspace(full_kspace, old_mask)  
        masks = {}      
        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            first_recon_abs = torch.norm(first_recon, dim=1, keepdim=True)  
            
            
            out_seg = self.seg_net(first_recon_abs)  

            if self.args.dataset_name == 'ACDC':
                if self.args.rate == 0.05:
                    
                    sec_input = torch.cat([first_recon, zero_recon], dim=1)  
                elif self.args.rate == 0.1 or self.args.maskType == '1D':
                    
                    sec_input = torch.cat([first_recon_abs, first_recon, zero_recon], dim=1)  
                else:
                    sec_input = first_recon
            else:
                sec_input = first_recon

            
            second_rec =self.fine_recon_net(sec_input, torch.sigmoid(out_seg))

            old_mask = new_mask
            pred_kspace = fft(second_rec)
            masks[i] = old_mask

        second_rec_abs = torch.norm(second_rec, dim=1, keepdim=True)  
        out_seg = self.seg_net(second_rec_abs)  
        second_rec =self.fine_recon_net(sec_input, torch.sigmoid(out_seg))

        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon, 'out_seg':out_seg, 'second_rec': second_rec, 
                    'masks':masks
                    }
        return pred_dict