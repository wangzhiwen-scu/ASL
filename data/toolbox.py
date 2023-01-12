import os
import numpy as np
import cv2
import torch
import glob
import h5py
import sys
sys.path.append('.') 

def crop_arr(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (h, w), size is target size.
    """
    h, w = arr.shape[0], arr.shape[1]
    th, tw = size[0], size[1]
    crop_img = arr[int(h/2)-int(th/2):int(h/2)+int(th/2), int(w/2)-int(tw/2):int(w/2)+int(tw/2)]
    return crop_img

def resize_vol_img(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, h, w), size is target size.
    """
    s, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    new_arr = np.zeros((s, size, size))
    for i in range(arr.shape[0]):
        new_arr[i] = cv2.resize(arr[i], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def resize_vol_seg(arr, size):
    """resize img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, channel,h, w), size is target size.
    """
    s, c, h, w = arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3]
    new_arr = np.zeros((s, c, size, size))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = cv2.resize(arr[i][j], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def get_filePath_fileName_fileExt(fileUrl):
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension

def get_mrb_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    if 'MRBrainS13DataNii' in filepath: 
        realshotname = filepath.split("/")[-1]
    elif '18training_corrected' in filepath:
        realshotname = filepath.split("/")[-2]
    
    return realshotname

def get_acdc_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    
    return realshotname

def get_braints_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    print(realshotname)
    return realshotname

def get_oai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    print(shotname)
    return shotname

def get_miccai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    
    return shotname

def get_h5py_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    print(shotname)
    return shotname

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def torch0to1(img):
    B, C, H, W = img.shape
    for b in range(B):
        img_min = torch.min(img[b, :, :,:])
        img_max = torch.max(img[b, :, :,:])
        img[b, :, :,:] = 1.0 * (img[b, :, :,:] - img_min) / (img_max - img_min)
    return img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


class OASI1_MRB(object):

    @staticmethod
    def get_oasi1mrb_edge_h5py():
        raw_path = r'./datasets/brain/OASI1_MRB/'
        
        
        train_img_path = glob.glob(raw_path+'testing-h5py/demo/*.h5')
        
        test_img_path = glob.glob(raw_path+'testing-h5py/demo/*.h5')

        return train_img_path, test_img_path
    
    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            edge = data['edge'][slice]
            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            edge = edge.astype(np.double)
            return img_, seg_, edge
