import os
import time
import sys
import warnings



import torch
from torch import nn



sys.path.append('.') 

from modeling.comb_net import CSMTLNet
from solver.loss_function import LocalStructuralLoss
from utils.train_utils import return_data_ncl_imgsize
warnings.filterwarnings("ignore")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class ModelSet():
    def __init__(self, args, test=False):
        '''

        '''
        cartesian = False
        module = ''
        if args.maskType == '1D':
            args.mask = 'cartesian'
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))

        self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)
        
        self.weight_recon, self.weight_seg, self.weight_local= 1, 0.1, 0.01 
        self.criterion = [nn.SmoothL1Loss(), nn.BCEWithLogitsLoss(), LocalStructuralLoss()]

        self.ckpt = './model_zoo/tab1/{}/csmtl_{}_{}.pth'.format(args.dataset_name, module, str(args.rate))
        args.save_path = self.ckpt
        self.model = CSMTLNet(args=args, nclasses=self.nclasses, inputs_size=self.inputs_size).to(device, dtype=torch.float)
        args.save_path = self.ckpt
        self.lr = args.lr  

        self.num_epochs = args.epochs  
        print('MTL weight_recon={}, weight_seg={}, weight_local={}'.format(self.weight_recon, self.weight_seg, self.weight_local))
        self.optimizer = torch.optim.Adam([
            {'params': self.model.recon_net.parameters()}], lr=self.lr)
        print('different lr0 = {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))

        patience = 100
        self.batch = args.batch_size
        self.save_epoch = 10  
        
        self.milestones = [200, 400]

        if test:
            self.model.load_state_dict(
                torch.load(args.save_path))
            print("Finished load model parameters! in {}".format(args.save_path))

       

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        else:
            print( 'Using '+ str(device))
        self.model.to(device, dtype=torch.float)

        scheduler_trick = 'ReduceLROnPlateau'
        if scheduler_trick == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        factor=0.1,
                                                        patience=patience, 
                                                        verbose=True,
                                                        threshold=0.0001,
                                                        min_lr=1e-8,
                                                        cooldown=4)
        elif scheduler_trick == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        elif scheduler_trick == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                milestones=self.milestones, gamma=0.1)            
        self.scheduler_trick = scheduler_trick
        print('scheduler_trick={}'.format(self.scheduler_trick))

        print('batch={}, lr={}, dataset_name={}, stage={}, load_model={}, save_per_epoch={}'\
            .format(self.batch, self.lr, args.dataset_name, args.stage, args.load_model, 'best-model'))
        if test:
            print('It just a test' + '*'*20)

    def recon_criterion(self, out_recon, full_img, seg_lable):
        rec_loss = self.criterion(out_recon, full_img)
        perceptual_loss = self.local_loss(out_recon, full_img, seg_lable, seg_lable) 
        loss = rec_loss + perceptual_loss * self.weight_local
        return loss


    def comb_valina_criterion(self, out_recon, full_img, out_seg, seg_lable): 
        loss_recon = self.criterion[0](out_recon, full_img)
        loss_seg = self.criterion[1](out_seg, seg_lable)
        
        loss = self.weight_recon * loss_recon + self.weight_seg * loss_seg
        return loss

    def comb_criterion(self, out_recon, full_img, out_seg, seg_lable, edge): 
        loss_recon = self.criterion[0](out_recon, full_img)
        loss_seg = self.criterion[1](out_seg, seg_lable)
        loss_local_recon = self.criterion[2](out_recon, full_img, edge, edge)
        loss = self.weight_recon * loss_recon + self.weight_seg * loss_seg + self.weight_local * loss_local_recon
        return loss
