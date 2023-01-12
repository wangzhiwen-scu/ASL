import sys
import numpy as np
import warnings
import random
import h5py

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

sys.path.append('.') 
from data.toolbox import get_h5py_shotname
from data.toolbox import OASI1_MRB

warnings.filterwarnings("ignore")


class H5PYMixedSliceData(Dataset): 
    def __init__(self, dataset_name, root=None, validation=False, test=False,seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
           
        
        if dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
        if validation:
            test_data = test_data[0:1]
            
        if root == None:
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data

        
        self.examples = []
        
        print('Loading dataset :', root)
        random.seed(seed)

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['img']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples += [(fname, shotname, slice) for slice in range(num_slices)] 

        if test:
            self.transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            ) 
            self.target_transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            )           
        else:
            self.transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, shotname, slice = self.examples[i]
        img_, seg_, edge = OASI1_MRB.getslicefromh5py(fname, slice)  

        y_train = seg_.transpose((1,2,0))  

        y_edge = edge

        
        x_train = img_.astype(np.float32) 

        y_train = y_train.astype(np.float32)
        y_edge = y_edge.astype(np.float32) 

        y_edge_ = y_edge[:,:, np.newaxis]
        y_train = np.concatenate((y_train, y_edge_), axis=-1) 

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        
        x_train = self.transform(x_train)
        random.seed(seed) 
        torch.manual_seed(seed) 
        y_train = self.transform(y_train)
        random.seed(seed) 
        torch.manual_seed(seed) 
        y_edge = self.transform(y_edge)
        return x_train, y_train, y_edge, shotname
