"""
    Mask Learning Module.
    Pytorch version.
    By WZW.
"""

import sys
import time


import torch
from torch import nn, optim

sys.path.append('.') 
from modeling.seg_net import SimpleUnet

from utils.train_utils import return_data_ncl_imgsize

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


class SegModelSet():
    def __init__(self, args, test=False):
        '''ckpt[0] for finetune, ckpt[1] for continue_train.
        mode must be 'recon', 'recon_finetune', 'comb', 'comn_finetune'
        '''
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))

        self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)

        self.name = 'seg_model'
        self.model = SimpleUnet(1, self.nclasses)
        self.ckpt = "./model_zoo/pretrained_seg/{}_{}seg.pth".format(args.dataset_name, self.nclasses)
        if args.test:

            
            self._get_weights(self.ckpt, self.model)
        
        self.batch = args.batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = args.lr
        self.num_epochs = args.epochs
        self.save_epoch = 10
        self.milestones = [100, 120, 140, 160, 180, 200]

        self.model.to(device, dtype=torch.float)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  
            
        scheduler_trick = 'ReduceLROnPlateau'
        if scheduler_trick == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        factor=0.1,
                                                        patience=5, 
                                                        verbose=True,
                                                        threshold=0.001,
                                                        min_lr=1e-9,
                                                        cooldown=4)
        elif scheduler_trick == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        self.scheduler_trick = scheduler_trick
        print('scheduler_trick={}'.format(self.scheduler_trick))

        print('batch={}, lr={}, dataset_name={}, save_per_epoch={}'\
            .format(self.batch, self.lr, args.dataset_name, 'best-model'))
        if test == True:
            print('It just a test' + '*'*20)

    def _get_weights(self, ckpt, model):
        pre_trained_model = torch.load(ckpt)
        new = list(pre_trained_model.items())
        my_model_kvpair = model.state_dict()
        count = 0
        for key, value in my_model_kvpair.items():
            layer_name, weights = new[count]
            my_model_kvpair[key] = weights
            count += 1
            
            
            
        model.load_state_dict(my_model_kvpair)    