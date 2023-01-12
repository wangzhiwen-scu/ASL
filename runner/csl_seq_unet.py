import os
import time
import sys
import warnings

import torch
from torch import nn

sys.path.append('.') 
from data.dataset import get_h5py_mixed_dataset as get_h5py_dataset
from utils.train_utils import return_data_ncl_imgsize
warnings.filterwarnings("ignore")
from modeling.endtoendtoolbox.samplenet_istanet import SequentialUnet

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
            cartesian = True

        if args.maskType == '1D':
            line_constrained=True
        elif args.maskType == '2D':
            line_constrained=False

        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
        self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)

        self.weight_recon, self.weight_seg, self.weight_local= 1, 0.1, 100  
        self.criterion = nn.SmoothL1Loss()
        num_step=3          

        if args.stage == '1':

            self.ckpt = './model_zoo/tab1/{}/csl_seqmri_unet_{}_{}_{}.pth'.format(args.dataset_name, module, str(args.rate), args.maskType)
            args.save_path = self.ckpt
            
            self.model = SequentialUnet(num_step=num_step, shape=[240,240], preselect=True, line_constrained=line_constrained, sparsity=args.rate, preselect_num=2, mini_batch=args.batch_size, 
                                        reconstructor_name='ista') 
            self.lr = args.lr  
            self.num_epochs = args.stage1_epochs  

            print('MTL weight_recon={}, weight_seg={}, weight_local={}'.format(self.weight_recon, self.weight_seg, self.weight_local))
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        base_learning_rate=5.5e-4 
        weight_decay=0.05
        warmup_epoch=5
        total_epoch=args.stage1_epochs
        batch_size = args.batch_size
        self.batch = args.batch_size
        self.save_epoch = 10  
        
        self.milestones = [200, 400]

        if test:
            self.model.load_state_dict(
                torch.load(args.save_path))
            print("Finished load model parameters! in {}".format(args.save_path))

        self.dataset, self.val_loader = get_h5py_dataset(args.dataset_name)


        

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        else:
            print( 'Using '+ str(device))
        self.model.to(device, dtype=torch.float)

        scheduler_trick = 'ReduceLROnPlateau'
        patience = 100
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                    factor=0.1,
                                                    patience=patience, 
                                                    verbose=True,
                                                    threshold=0.0001,
                                                    min_lr=1e-8,
                                                    cooldown=4)
        
        self.scheduler_trick = scheduler_trick

        print('scheduler_trick={}'.format(self.scheduler_trick))

        print('batch={}, lr={}, dataset_name={}, stage={}, load_model={}, save_per_epoch={}'\
            .format(self.batch, self.lr, args.dataset_name, args.stage, args.load_model, 'best-model'))
        if test:
            print('It just a test' + '*'*20)

    def recon_criterion(self, out_recon, full_img):
        rec_loss = self.criterion(out_recon, full_img)
        return rec_loss

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