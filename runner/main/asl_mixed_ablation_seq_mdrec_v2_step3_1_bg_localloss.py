import os
import time
import sys
import warnings
import argparse



import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
sys.path.append('.') 

from utils.visdom_visualizer import VisdomLinePlotter
from utils.train_metrics import Metrics
warnings.filterwarnings("ignore")

from data.dataset import get_h5py_mixed_dataset as get_h5py_dataset
from modeling.comb_unet2_1_localloss import SequentialASLDualDo as SequentialASL

from utils.train_metrics import Metrics, EDiceLoss, dice_coef, LocalStructuralLoss
from utils.train_utils import return_data_ncl_imgsize, update_partial_dict

from utils.logger import Logger

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
            line_constrained=True
        elif args.maskType == '2D':
            line_constrained=False

        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
        self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)
        self.nclasses += 1
        self.nclasses += 1 

        self.weight_recon, self.weight_seg, self.weight_local= 1, 0.1, 100  

        if args.stage == '1':

            self.ckpt = './model_zoo/tab1/{}/asl_ablation_seqmdrecnet_bg_step3_1_local_{}_{}_{}.pth'.format(args.dataset_name, module, str(args.rate), args.maskType)
            
            args.save_path = self.ckpt
            
            self.model = SequentialASL(num_step=3, shape=[240,240], preselect=True, line_constrained=line_constrained, sparsity=args.rate, 
                                        preselect_num=2, mini_batch=args.batch_size, inputs_size=self.inputs_size, nclasses=self.nclasses, args=args)   
            self.lr = args.lr  
            self.num_epochs = args.stage1_epochs  

            
            self.stage0_ckpt = './model_zoo/tab1/{}/asl_ablation_seqmdrecnet_bg_step2_1_local_{}_{}_{}.pth'.format(args.dataset_name, module, str(args.rate), args.maskType)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            if not test:
                pretrained_dict = torch.load(self.stage0_ckpt)
                update_partial_dict(pretrained_dict, self.model)

            if args.maskType == '2D':
                for param in self.model.sampler.parameters():
                    param.requires_grad = False

            for param in self.model.reconstructor.parameters():
                param.requires_grad = False

            for param in self.model.seg_net.parameters(): 
                param.requires_grad = False

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
        
        self.scheduler_trick = scheduler_trick
        patience = 100
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                    factor=0.1,
                                                    patience=patience, 
                                                    verbose=True,
                                                    threshold=0.0001,
                                                    min_lr=1e-8,
                                                    cooldown=4)
        print('scheduler_trick={}'.format(self.scheduler_trick))

        print('batch={}, lr={}, dataset_name={}, stage={}, load_model={}, save_per_epoch={}'\
            .format(self.batch, self.lr, args.dataset_name, args.stage, args.load_model, 'best-model'))
        if test:
            print('It just a test' + '*'*20)

        self.criterion = [nn.SmoothL1Loss(), nn.BCEWithLogitsLoss(), EDiceLoss(), LocalStructuralLoss()]

        if args.maskType == '2D':
            self.var_lambda = 0.96
            self.var_local = 5*(1-self.var_lambda)
            
        elif args.maskType == '1D':
            self.var_lambda = 0.9
            self.var_local = 0.99
            if args.dataset_name == 'ACDC':
                self.var_lambda = 0.1
                self.weight_seg = 100
                self.var_local = 0.001*(1-self.var_lambda)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate") 
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The batch size")
    parser.add_argument(
        "--stage1_epochs", type=int, default=600, help="The weighted parameter") 
    parser.add_argument(
        "--stage2_epochs", type=int, default=600, help="The weighted parameter") 
    parser.add_argument(
        "--rate",
        type=float,
        default=0.01,
        choices=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25],
        help="The undersampling rate")
    parser.add_argument(
        "--mask",
        type=str,
        default="random",
        choices=["cartesian", "radial", "random"],
        help="The type of mask")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="which dataset")
    parser.add_argument(
        "--load_model",
        type=int,
        default=0,
        choices=[0,1],
        help="reload last parameter?")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=['1','2'],
        help="1: coarse rec net; 2: fixed coarse then train seg; 3: fixed coarse and seg, then train fine; 4: all finetune.")
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        choices=[True, False])
    parser.add_argument(
        "--maskType",
        type=str,
        default='2D',
        choices=['1D', '2D'])
    args = parser.parse_args()