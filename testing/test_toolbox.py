import sys
import os

import torch
from torch.autograd import Variable
import numpy as np
from numpy.fft import fftshift
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import cv2
from matplotlib.image import imsave
from utils.train_metrics import dice_coef
import pandas as pd
from PIL import Image


sys.path.append('.') 
from modeling.dualdomain import MRIReconstruction
from layers.mask_layer import Mask_Fixed_CartesianLayer, Mask_Fixed_Layer
from utils.train_utils import return_data_ncl_imgsize


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def fakedataloder(input, mask_torch):

    label_img = input[0, 0, ...].detach().cpu().numpy()
    
    
    

    
    full_img = torch.zeros(*label_img.shape, 2)
    full_img[:, :, 0] = torch.from_numpy(label_img.real)
    full_img[:, :, 1] = torch.from_numpy(label_img.imag)
    full_img = full_img.cuda()

    
    full_k = torch.fft(full_img, 2, normalized=False)
    tmp = full_k.permute(2, 0, 1)
    under_k = tmp * mask_torch

    tmp = under_k.permute(1, 2, 0)
    under_img = torch.ifft(tmp, 2, normalized=False)

    full_k = full_k.permute(2, 0, 1)
    full_img = full_img.permute(2, 0, 1)
    under_img = under_img.permute(2, 0, 1)
    return under_img, under_k, full_img, full_k

def createinput(f_img, mask_torch):
    """ f_img.shape = (1, 1, H, W)
    """
    under_img, under_k, full_img, full_k = fakedataloder(f_img, mask_torch)
    under_img = torch.unsqueeze(under_img, dim=0)
    under_k = torch.unsqueeze(under_k, dim=0)
    full_img = torch.unsqueeze(full_img, dim=0)
    full_k = torch.unsqueeze(full_k, dim=0)

    u_img = Variable(under_img, requires_grad=False).cuda()
    u_k = Variable(under_k, requires_grad=False).cuda()
    img = Variable(full_img, requires_grad=False).cuda()
    k = Variable(full_k, requires_grad=False).cuda()
    return u_img, u_k, img, k

def mergeMultiLabelToOne(segs):
    """Inputs: [n, seg_x, seg_y], n: how many segmentation.

    Args:
        segs (_type_): _description_
    """
    new_seg = np.zeros((segs.shape[1], segs.shape[2]))
    for i in range(segs.shape[0]):
        new_seg[segs[i]==1] =1
    return new_seg

def mask_find_bboxs(mask):
    """
    """
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) 
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] 

def get_roi(imgs_gt, imgs_key, segs_gt):
    """https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html


    -------------------------------------------
    |                                         | 
    |    (x1, y1)                             |
    |      ------------------------           |
    |      |                      |           |
    |      |                      |           | 
    |      |         ROI          |           |  
    |      |                      |           |   
    |      |                      |           |   
    |      |                      |           |       
    |      ------------------------           |   
    |                           (x2, y2)      |    
    |                                         |             
    |                                         |             
    |                                         |             
    -------------------------------------------
    Args:
        imgs_gt (_type_): _description_
        imgs_key (_type_): _description_
        segs_gt (_type_): _description_
    """
    mask = mergeMultiLabelToOne(segs_gt)
    bin_uint8 = (mask * 255).astype(np.uint8)
    bboxs = mask_find_bboxs(bin_uint8)
    
    try:
        b = bboxs[0]
        x1, y1 = b[0], b[1]
        x2 = b[0] + b[2]
        y2 = b[1] + b[3]
        
        
        ROI_gt = imgs_gt[y1:y2, x1:x2]
        ROI_key = imgs_key[y1:y2, x1:x2]
        return ROI_gt, ROI_key
    except IndexError:
        return imgs_gt, imgs_key

class Dual_model():
    def __init__(self, args, acc, mask_type):
        model_type = "model"
        data_type = "brain"
        
        w = 0.2 
        bn = False
        
        __nclasses, img_size = return_data_ncl_imgsize(args.dataset_name)
        
        module=''
        pre_path = "./data/masks"
        mask = sio.loadmat(pre_path + "/{}_{}_{}_{}.mat".format(mask_type, img_size[0], img_size[1], acc))['Umask']  
        
        path = './model_zoo/tab1/{}/csmri2__{}.pth'.format(args.dataset_name, int(args.rate*100))
        mask = fftshift(mask, axes=(-2, -1))
        
        print(path)
        
        mask_torch = torch.from_numpy(mask).float().cuda()
        model = MRIReconstruction(mask_torch, w, bn).cuda()
        if os.path.exists(
                path):
            model.load_state_dict(
                torch.load(path))
            print("Finished load model parameters!")
        self.model = model
        self.mask_torch = mask_torch

def plotRoiInImg(img, data_name, namespace=None):
    
    scale = 2
    roiBox = {}
    
    roiBox['ACDC']=(70, 100, 70+30, 100+30)  
    
    roiBox['BrainTS']=(70, 100, 70+30, 100+30)  
    
    
    roiBox['OAI']=(190, 170, 190+50, 170+50)
    roiBox['MRB']=(130, 100, 130+50, 100+50) 
    roiBox['MICCAI']=(130, 100, 130+50, 100+50) 
    roiBox['OASI1_MRB']=(130, 100, 130+50, 100+50) 
    roiBox['fastMRI']=(130, 110, 130+50, 110+50) 
    roiBox['Mms']=(70, 100, 70+30, 100+30)  
    roiBox['Prostate']=(130, 100, 130+50, 100+50) 
    
    (x1, y1, x2, y2) = roiBox[data_name]

    color1 = (255, 0, 0)

    img2 = img[..., np.newaxis].astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  
    roi = img2[y1:y2, x1:x2]  
    roi2 = cv2.resize(roi, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    
    roi2 = cv2.rectangle(roi2, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)

    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)
    
    
    img2[0:scale*(x2-x1), -scale*(y2-y1)-1:-1] = roi2
    
    return img2

def plotRoiOutImgTwo(img, channles=1, error_map=False, namespace=None):
    
            
        
        
        
        
        
        
        

    scale = 2
    roiBox = {}
    
    
    
    
    
    
    
    
    
    
    
    if namespace:
        roiBox['OASI1_MRB1']=namespace.bbox1 
        roiBox['OASI1_MRB2']=namespace.bbox2   
    else:
        roiBox['OASI1_MRB1']=(128, 100, 128+60, 100+30) 
        roiBox['OASI1_MRB2']=(95, 170, 95+60, 170+30)   
    
    
    
    

    color1 = 	(255,0,0) 
    color2 = 	(255,240,0)

    
    

    
    (x1, y1, x2, y2) = roiBox['OASI1_MRB1']

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)
    
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    
    (x1, y1, x2, y2) = roiBox['OASI1_MRB2']
    roi2 = img2[y1:y2, x1:x2]  
    roi22 = cv2.resize(roi2, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi22 = cv2.rectangle(roi22, (0+1, 0+1), (scale*(x2-x1)-2, scale*(y2-y1)-2), color2, thickness=2, lineType=cv2.LINE_AA)

    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color2, thickness=2, lineType=cv2.LINE_AA)
    
    

    
    
    return img2, roi11, roi22

def plotRoiOutImgACDC(img, channles=1, namespace=None, error_map=False):
    
            
        
        
        
        
        
        
        

    scale = 2
    roiBox = {}
    
    if namespace:
        roiBox['ACDC'] = namespace.bbox
    else:
        roiBox['ACDC']=(50, 90, 50+120, 90+60)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    color1 = 	(255,0,0) 
    color2 = (255,255,0)  

    
    

    
    (x1, y1, x2, y2) = roiBox['ACDC']

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)
    
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    return img2, roi11

def plotRoiOutImgProstate(img, channles=1, error_map=False, namespace=None):
    
            
        
        
        
        
        
        
        

    scale = 2
    roiBox = {}
    
    if namespace:
        roiBox['Prostate'] = namespace.bbox
    else:
        roiBox['Prostate']=(70, 80, 70+120, 80+60)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    color1 = 	(255,0,0) 
    color2 = 	(255,240,0)

    
    (x1, y1, x2, y2) = roiBox['Prostate']

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)
    
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    return img2, roi11

def save_org_for_paper(args, imgs, segs, masks, patient_name, data_name, desired_sparsity):

    error_maps = {}
    
    error_v = {"ANDI": (0, 100), "Mms": (0, 100), 'OAI':(0, 60), "MRB": (0, 100)}
    vmin, vmax = error_v[data_name]
    
    
    
    for key in imgs.keys():
        img_temp = imgs[key]
        _img_min = np.min(img_temp)
        _img_max = np.max(img_temp)
        imgs[key] = 255.0 * (img_temp - _img_min) / (_img_max - _img_min)
    for key in imgs.keys():
        if key == 'gt':
            continue
        error_maps[key] = np.abs(imgs['gt'] - imgs[key])    
    
    
    save_dir = args.PREFIX + 'sparsity_{}/{}/' .format(desired_sparsity, patient_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(save_dir + " created successfully.")

    for key in imgs.keys():
        
        _img = imgs[key]
        
        save_path = save_dir + key + '.png'
        imsave(save_path, _img, dpi=300, vmin=0, vmax=100)
        if key == 'gt':
            continue
        imsave(save_dir + key + '_ermap.png', error_maps[key], dpi=300, vmin=vmin, vmax=vmax)

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + " created successfully.")

    
def get_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 
                'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1), 'fastMRI':(0,1), 'Mms':(0,1)}
    vmin, vmax = error_v[data_name]
    
    

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) 

    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    
    ax = axes_flt[0]
    
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        
        
        
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        
        
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        
        
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  

    
    
    
    
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) 
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    
    
    
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    
    
    
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_backbone_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1)}
    vmin, vmax = error_v[data_name]
    
    

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) 

    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    
    ax = axes_flt[0]
    
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr')
    
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        
        
        
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        
        
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        
        
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  

    
    
    
    
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) 
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    
    
    
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    
    
    
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_fastmri_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 
                'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1), 'fastMRI':(0,1), 'Mms':(0,1)}
    vmin, vmax = error_v[data_name]
    
    

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) /255.0 

    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    
    ax = axes_flt[0]
    
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        
        
        
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        
        
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        
        
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  

    
    
    
    
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) 
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    
    
    
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def get_fastmri_onlyone_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    
    
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    

    error_maps = {}
    max_rangs = {}
    error_maps_offset ={}

    vmin, vmax = (0, 1)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    for key in keyssss:
        max_rangs[key] = np.mean(imgs[key])-np.min(imgs[key])
        

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) /255.0 

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec = plotRoiInImg(img4plot(imgs[key]), data_name='fastMRI', namespace=namespace)

        
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        
        ax = plt.subplot2grid((nrows, ncols), (2, now_col), rowspan=2, colspan=1)
        
        
        

        ax.set_axis_off()

        if key != 'gt':
        
            im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
            
        if key == 'csmri1':
            cb_ax = fig.add_axes([0.15, 0.21, 0.01, 0.38]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()
            

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_fastmri_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def open_morph(imgs):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(imgs, cv2.MORPH_OPEN, kernel)

    for i in range(imgs.shape[0]):
        threshold =np.max(imgs[i]) - np.min(imgs[i]) / 2
        imgs[i][imgs[i] >= threshold] = 1
        imgs[i][imgs[i] < threshold] = 0

    return opening

def tensor2np(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    
    img = np.squeeze(img)
    
    
    
    return img

def img4plot(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    
    img = np.squeeze(img)
    min = np.min(img) 
    max = np.max(img) 
    img = 255.0 * ((img - min)/ (max - min)) 
    return img

def normalize(img, max_int=255.0):
    """ normalize image to [0, max_int] according to image intensities [v_min, v_max] """
    v_min, v_max = np.max(img), np.min(img)
    img = (img - v_min)*(max_int)/(v_max - v_min)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img    

def tensor2np4soloseg(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    img = np.squeeze(img)
    return img    

def seg2np(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    img = np.abs(np.squeeze(img))
    
    
    
    return img

def save_mask_as_np(args, data_name, traj_type, mode, ckpt, desired_sparsity):
    path = {}
    zero_one_np = {}
    
    
    
    
    
    

    path['unet_mask'] = args.save_path1
    path['mtl1_mask'] = args.save_path

    img = torch.randn((1, 1, 240, 240)).to(device)
    for key in path.keys():
        if traj_type == "cartesian":
            mask_layer = Mask_Fixed_CartesianLayer(ckpt=path[key], desired_sparsity=desired_sparsity).to(device)
        elif traj_type == "radial":
            pass
        elif traj_type == "random":
            mask_layer = Mask_Fixed_Layer(ckpt=path[key], desired_sparsity=desired_sparsity).to(device)

        (uifft, complex_abs, zero_one, fft, undersample) = mask_layer(img)
        
        zero_one_np[key] = img4plot(zero_one) / 255.0
        zero_one_np[key] = np.fft.fftshift(zero_one_np[key])
        print("undersampling_rate is {}".format(np.mean(zero_one_np[key])))
        
        save_path = args.PREFIX + 'sparsity_{}/{}.png' .format(desired_sparsity, key)
        imsave(save_path, zero_one_np[key], cmap='gray')
    return zero_one_np

def get_metrics(args, imgs, segs, df, traj_type, shotname, NUM_SLICE, desired_sparsity):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    
    for key in segs.keys():
        temp_df = init_temp_df(dataset_name=args.dataset_name)
        if key == 'gt':
            continue
        
        
        max_value = np.max(imgs['gt'])-np.min(imgs['gt'])
        if key == 'supre':
            temp_df['PSNR'] = None
            temp_df['SSIM'] = None
        else:
            temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key], data_range=max_value), 2) 
            temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key], data_range=max_value), 2) 
        for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  
            temp_dice = dice_coef(segs[key][i], segs['gt'][i])
            temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  
            
        

        
        
        
        
        
        
        
        
        

        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)

def get_ROI_metrics(args, imgs, segs, df, traj_type, shotname, NUM_SLICE, desired_sparsity):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    for key in imgs.keys():
        temp_df = init_temp_df(dataset_name=args.dataset_name)
        if key == 'gt':
            continue
        
        
        
        
        
        
        
        
        
        

        
        ROI_gt, ROI_key = get_roi(imgs['gt'], imgs[key], segs['gt'])  
        max_value = np.max(ROI_gt)-np.min(ROI_gt)
        temp_df['PSNR'] = round(compare_psnr(ROI_gt, ROI_key, data_range=max_value), 2) 
        temp_df['SSIM'] = round(100*compare_ssim(ROI_gt, ROI_key, data_range=max_value), 2) 
        for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  
            temp_dice = dice_coef(segs[key][i], segs['gt'][i])
            temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  
        

        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)


def get_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], 
                                    [0, 0, 255], 
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        
        
        
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        ax = axes_flt[num_plot+6]
        
        
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)

        
        
        
        
        
            

        
        
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_seg_oasis1_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 10, 6
    
    
    ax = {}
    fig = plt.figure(figsize=(19, 15))
    

    error_maps = {}
    error_maps_offset = {}
    max_rangs = {}

    vmin, vmax = (0, 1)
    
    
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    
    for key in keyssss:
        
            
        max_rangs[key] = np.mean(imgs[key])-np.min(imgs[key])



        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 
        

    
    for key in keyssss:
        max_max_range = max(zip(max_rangs.values(), max_rangs.keys()))  
        error_maps_offset[key] = max_rangs[key]/max_max_range[0]
        error_maps[key] = img4plot(error_maps[key]) /255.0 

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    
    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        

        
        
        
        
        
        
        
        

        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.3 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],

                                    
                                    
                                    

                                    [153,153,153],
                                    [188,188,188],
                                    [238,238,238],

                                    
                                    
                                    
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        
        
        
        region = img_segmaps_np  

        now_col = num_plot

        
        

        

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        
        

        img_rec, roi1, roi2 = plotRoiOutImgTwo(img4plot(imgs[key]), namespace=namespace)

        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (2, now_col), rowspan=2, colspan=1)
        
        
        segmap, segroi1, segroi2 = plotRoiOutImgTwo(region, channles=3, namespace=namespace)
        im = ax.imshow(segmap)
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (4,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi1, 'gray')
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (5,now_col), rowspan=1, colspan=1)
        if key != 'gt':
            _img, errorroi1, errorroi2 = plotRoiOutImgTwo(img4plot(error_maps[key]), error_map=True, namespace=namespace)
            errorroi1 = img4plot(cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)) /255.0
            
            
            if key == 'asl':
                im = ax.imshow(errorroi1 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
            else:
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='jet')

        if key == 'asl':
            cb_ax = fig.add_axes([0.16, 0.407, 0.004, 0.085]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (6,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi1)
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (7,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi2, 'gray')
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (8,now_col), rowspan=1, colspan=1)
        if key != 'gt':
            errorroi2 = img4plot(cv2.cvtColor(errorroi2, cv2.COLOR_RGB2GRAY)) /255.0
            im = ax.imshow(errorroi2 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
        if key == 'csmri1':
            
            cb_ax = fig.add_axes([0.16, 0.108, 0.004, 0.085]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()
            
        ax.set_axis_off()
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (9,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi2)
        ax.set_axis_off()
        

        
        
        num_plot = num_plot + 1
        

    save_path = args.PREFIX + 'figs_{}_{}_{}_seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    
    
    
    
    
    
    

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_masks_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    
    
    ax = {}
    fig = plt.figure(figsize=(9, 4))

    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    
    

    key = 'csmri1'
    ax = plt.subplot2grid((nrows, ncols), (0,0), rowspan=1, colspan=1)
    
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    key = 'csl'
    ax = plt.subplot2grid((nrows, ncols), (0,1), rowspan=1, colspan=3)
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    key = 'asl'
    ax = plt.subplot2grid((nrows, ncols), (0,4), rowspan=1, colspan=3)
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    save_path = args.PREFIX + 'mask/figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_masks_map_onebyone_gridplot(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    
    
    
    fig = plt.figure(figsize=(10, 19))

    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    
    

    key = 'csmri1'
    
    
    masks_plot = np.fft.fftshift(img4plot(masks[key]))

    ax = plt.subplot2grid((nrows, ncols), (0,0), rowspan=1, colspan=1)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()
    
    

    key = 'csl'
    for i in range(3):
        
        masks_now = np.fft.fftshift(img4plot(masks[key][i]))

        if i == 0:
            masks_plot = masks_now
            
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    ax = plt.subplot2grid((nrows, ncols), (0,1), rowspan=1, colspan=3)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()

    key = 'asl'
    
    
    
    
    
    for i in range(3):
        masks_now = np.fft.fftshift(img4plot(masks[key][i]))
        if i == 0:
            masks_plot = masks_now
            
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    ax = plt.subplot2grid((nrows, ncols), (0,4), rowspan=1, colspan=3)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()


    save_path = args.PREFIX + 'mask/figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    
    plt.subplots_adjust(top = 1, bottom = 0, right = 0.4, left = 0, 
                hspace = 0, wspace = 0)
    

    
    
    
    plt.gcf().tight_layout()
    plt.savefig(save_path, 
    bbox_inches = 'tight',
        pad_inches = 0
        )

def get_masks_map_onebyone(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    
    
    
    
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10,19), gridspec_kw={'width_ratios': [1, 3, 3]})


    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    
    

    f_k = np.abs(np.fft.fft2(imgs['gt']))
    f_k = np.log(f_k)

    key = 'csmri1'
    mask_csmri1= img4plot(masks[key])/255.0
    mask_csmri1 = np.rot90(mask_csmri1)
    
    masks_now = np.fft.fftshift(mask_csmri1*f_k)
    masks_now = img4plot(masks_now)
    a0.imshow(masks_now)
    a0.set_axis_off()
    
    

    

    key = 'csl'
    for i in range(3):
        
        masks_now = np.fft.fftshift(img4plot(masks[key][i])/255.0*f_k)
        masks_now = img4plot(masks_now)

        if i == 0:
            masks_plot = masks_now
            
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    
    a1.imshow(masks_plot)
    a1.set_axis_off()

    key = 'asl'
    
    
    
    
    
    for i in range(3):
        
        masks_now = np.fft.fftshift(img4plot(masks[key][i])/255.0*f_k)
        masks_now = img4plot(masks_now)

        if i == 0:
            masks_plot = masks_now
            
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    a2.imshow(masks_plot)
    a2.set_axis_off()


    save_path = args.PREFIX + 'figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.subplots_adjust(top = 1, bottom = 0, right = 0.4, left = 0, 
                hspace = 0, wspace = 0)   
    plt.gcf().tight_layout()
    plt.savefig(save_path, 
    bbox_inches = 'tight',
        pad_inches = 0
        )

def get_backbone_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(19, 8))
    
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr', 'asl_mdr')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], 
                                    [0, 0, 255], 
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        
        
        
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        
        ax = axes_flt[num_plot+7]
        
        
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)

        
        
        
        
        
            

        
        
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_seg_acdc_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    
    
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    

    error_maps = {}
    error_maps_offset = {}
    max_rangs = {}

    vmin, vmax = (0, 1)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    for key in keyssss:
        
        max_rangs[key] = np.max(imgs[key])-np.min(imgs[key])

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 

    for key in keyssss:
        max_max_range = max(zip(max_rangs.values(), max_rangs.keys()))  
        

        if key == 'asl':
            error_maps_offset[key] = max_rangs[key]/max_max_range[0]

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        
        
        
        
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.5 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],

                                    
                                    
                                    

                                    [85,85,85],
                                    [170,170,170],
                                    [255,255,255],

                                    
                                    
                                    
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        region = img_segmaps_np  

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec, roi1 = plotRoiOutImgACDC(img4plot(imgs[key]), namespace=namespace)

        
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        
        
        segmap, segroi1 = plotRoiOutImgACDC(region, channles=3, namespace=namespace)
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (2,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi1, 'gray')
        ax.set_axis_off()

        
        ax = plt.subplot2grid((nrows, ncols), (3,now_col), rowspan=1, colspan=1)
        ax.set_axis_off()
        if key != 'gt':
            _img, errorroi1 = plotRoiOutImgACDC(img4plot(error_maps[key]), error_map=True, namespace=namespace)
            errorroi1 = img4plot(cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)) /255.0
            
            
            if key == 'asl':
                im = ax.imshow(errorroi1 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
            else:
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='jet')
        if key == 'asl':
            cb_ax = fig.add_axes([0.16, 0.21, 0.004, 0.18]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()
            

        
        ax = plt.subplot2grid((nrows, ncols), (4,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi1)
        ax.set_axis_off()
        
        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_seg_prostate_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    
    
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    

    error_maps = {}
    error_maps_offset = {}
    max_rangs = {}

    vmin, vmax = (0, 1)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    for key in keyssss:
        max_rangs[key] = np.max(imgs[key])-np.min(imgs[key]) 

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) 
    for key in keyssss:
        max_max_range = max(zip(max_rangs.values(), max_rangs.keys()))  
        error_maps_offset[key] = max_rangs[key]/max_max_range[0]
        error_maps[key] = img4plot(error_maps[key]) /255.0 

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        
        
        
        
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.5 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],
                                    [85,85,85],
                                    [170,170,170],
                                    [255,255,255],

                                    
                                    
                                    
                                    
                                    
                                    
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        region = img_segmaps_np  

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec, roi1 = plotRoiOutImgProstate(img4plot(imgs[key]), namespace=namespace)

        
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        
        
        segmap, segroi1 = plotRoiOutImgProstate(region, channles=3, namespace=namespace)
        
        

        
        ax = plt.subplot2grid((nrows, ncols), (2,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi1, 'gray')
        ax.set_axis_off()

        
        ax = plt.subplot2grid((nrows, ncols), (3,now_col), rowspan=1, colspan=1)
        ax.set_axis_off()
        if key != 'gt':
            _img, errorroi1 = plotRoiOutImgProstate(img4plot(error_maps[key]), error_map=True, namespace=namespace)
            errorroi1 = img4plot(cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)) /255.0
            
            
            if key == 'asl':
                im = ax.imshow(errorroi1 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
            else:
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='jet')
        if key == 'asl':
            cb_ax = fig.add_axes([0.16, 0.21, 0.004, 0.18]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()

        
        ax = plt.subplot2grid((nrows, ncols), (4,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi1)
        ax.set_axis_off()
        
        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_ablation_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    
    
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(19, 8))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'woSegNet', 'woCRec','woSAM', 'woBG', 'woCLS', 'ASL')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], 
                                    [0, 0, 255], 
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        
        
        
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        
        ax = axes_flt[num_plot+7]
        
        
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)
       
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_ablation_seg_one_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 2, 7
    
    
    ax = {}
    fig = plt.figure(figsize=(16, 4.5))
    

    
    
    

    
    
    
    
    

    
    
    

    
    
    subplot_name_list = ('Ground truth', 'woSegNet', 'woCRec','woSAM', 'woBG', 'woCLS', 'ASL')
    
    num_plot = 0

    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.3 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],
                                    
                                    
                                    

                                    
                                    
                                    

                                    
                                    
                                    

                                    [198, 70, 70], 
                                    [120, 247, 120], 
                                    [21, 21, 149], 

                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        
        
        
        region = img_segmaps_np  

        now_col = num_plot

        

        img_rec = img4plot(imgs[key])
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=1, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        
        ax = plt.subplot2grid((nrows, ncols), (1, now_col), rowspan=1, colspan=1)
        
        
        segmap = region
        im = ax.imshow(segmap)
        ax.set_axis_off()

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_ablatiom_module_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_ablation_seg_one_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 2, 7
    
    
    ax = {}
    fig = plt.figure(figsize=(16, 4.5))
    

    
    
    

    
    
    
    
    

    
    
    

    
    
    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr', 'asl_mdr')
    
    num_plot = 0

    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.3 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],

                                    [198, 70, 70], 
                                    [120, 247, 120], 
                                    [21, 21, 149], 

                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        
        
        
        region = img_segmaps_np  

        now_col = num_plot

        

        img_rec = img4plot(imgs[key])
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=1, colspan=1)
        
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        
        ax = plt.subplot2grid((nrows, ncols), (1, now_col), rowspan=1, colspan=1)
        
        
        segmap = region
        im = ax.imshow(segmap)
        ax.set_axis_off()

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_ablation_backbone_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def get_ablation_step_seg_map(args, imgs, segs, masks, zero_recons, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    
    
    ncols = 4
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(16, 16))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'Step1', 'Step2', 'Step3')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], 
                                    [0, 0, 255], 
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        
        
        

        
        ax = axes_flt[num_plot+ncols*0]
        if key == 'gt':
            pass
        else:
            shift_mask = np.fft.ifftshift(masks[key])
            
                
            im = ax.imshow(img4plot(shift_mask), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        
        ax = axes_flt[num_plot+ncols*1]
        if key == 'gt':
            pass
        else:
            im = ax.imshow(img4plot(zero_recons[key]), 'gray')
        ax.set_axis_off()
        
        
        ax = axes_flt[num_plot+ncols*2]
        region = imgnp

        im = ax.imshow(img4plot(imgs[key]), 'gray')
        ax.set_axis_off()

        
        
        ax = axes_flt[num_plot+ncols*3]
        
        
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)
       
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_fig1_map(args, imgs, segs, masks, zero_recons,masked_kspaces, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    
    
    
    
    

    subplot_name_list = ('Ground truth', 'Step1', 'Step2', 'Step3')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        
        
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],

                                    [85,85,85],
                                    [255,255,255],
                                    [170,170,170],
                                    
                                    
                                    
                                    
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        
        
        
        dpi = 200
        
        if key == 'gt':
            pass
        else:
            shift_mask = np.fft.ifftshift(masks[key])

            save_path = args.PREFIX + 'mask_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(shift_mask), cmap='gray', dpi=dpi)


        
        if key == 'gt':
            pass
        else:
            save_path = args.PREFIX + 'zf_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(zero_recons[key]), cmap='gray', dpi=dpi)

        
        if key == 'gt':
            pass
        else:
            
            
            
            
            
            

            
            epsilon = 1e-8
            u_k = np.log(np.fft.ifftshift(masked_kspaces[key] + epsilon))
            

            save_path = args.PREFIX + 'uk_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(u_k), dpi=dpi)

        
        save_path = args.PREFIX + 'rec_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
        imsave(save_path, img4plot(imgs[key]), cmap='gray', dpi=300)

        
        save_path = args.PREFIX + 'seg_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
        imsave(save_path, imgnp, dpi=dpi)

        num_plot = num_plot + 1




def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + " created successfully.")

def newdf2excel(file_name, df):
    with pd.ExcelWriter(file_name) as writer:
        for key in df.keys():
            df[key].to_excel(writer, sheet_name=key) 

def dataframe_template(dataset_name):
    df_general = {"PSNR": [], "SSIM": []}
    if dataset_name == 'MRB':
        df_specific_dataset = {"Cortical GM": [], "Basal ganglia": [], "WM": [], "WM lesions": [], "CSF": [], "Ventricles": [], "Cerebellum": [], "Brainstem": []}  
    elif dataset_name == 'OAI':
        df_specific_dataset = {"Femoral Cart.": [], "Medial Tibial Cart.": [], "Lateral Tibial Cart.": [], "Patellar Cart.": [], "Lateral Meniscus": [], "Medial Meniscus": []}
    elif dataset_name == 'ACDC':
        df_specific_dataset = {"LV": [], "RV": [], "MYO": []}
    elif dataset_name == 'BrainTS':
        
        
        df_specific_dataset = {"NCR/NET": [], "ED": [], "ET": []}
    elif dataset_name == "MICCAI":
        df_specific_dataset = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7":[], "8": [], "9": [], "10": [],
                                "11":[], "12":[], "13":[], "14": [], "15":[], "16": [], "17": [], "18": [], "19": [], "20": [],
                                "21": [], "21": [], "22":[], "23": [], "24":[], "25":[], "26":[], "27":[], "28":[], "29":[], "30":[],
                                "31": [], "32":[], "33":[]}
    elif dataset_name == 'OASI1_MRB':
        df_specific_dataset = {"GM": [], "WM": [], "CSF": []}

    elif dataset_name == 'Prostate':
        df_specific_dataset = {"Seg1": [], "Seg2": []}
    df_last = {"shotname": [], "slice": []} 
    df = {**df_general, **df_specific_dataset, **df_last}
    return df

def init_temp_df_drop():
    temp_df = {"PSNR": None, "SSIM": None, "Cortical GM": None, "Basal ganglia": None, "WM": None, "WM lesions": 0,  
    "CSF": None,
    "Ventricles": None,
    "Cerebellum": None,
    "Brainstem": None
    }
    return temp_df

def init_temp_df(dataset_name):
    df_general = {"PSNR": None, "SSIM": None}
    if dataset_name == 'MRB':
        df_specific_dataset = {"Cortical GM": None, "Basal ganglia": None, "WM": None, "WM lesions": None, "CSF": None, "Ventricles": None, "Cerebellum": None, "Brainstem": None}  
    elif dataset_name == 'OAI':
        df_specific_dataset = {"Femoral Cart.": None, "Medial Tibial Cart.": None, "Lateral Tibial Cart.": None, "Patellar Cart.": None, "Lateral Meniscus": None, "Medial Meniscus": None}
    elif dataset_name == 'ACDC':
        df_specific_dataset = {"LV": None, "RV": None, "MYO":None}
    elif dataset_name == 'Mms':
        df_specific_dataset = {"LV": None, "RV": None, "MYO":None}
    elif dataset_name == 'BrainTS':
        
        
        df_specific_dataset = {"NCR/NET": None, "ED": None, "ET": None}
    elif dataset_name == "MICCAI":
        df_specific_dataset = {"1": None, "2": None, "3": None, "4": None, "5": None, "6": None, "7":None, "8": None, "9": None, "10": None,
                                "11":None, "12":None, "13":None, "14": None, "15":None, "16": None, "17": None, "18": None, "19": None, "20": None,
                                "21": None, "21": None, "22":None, "23": None, "24":None, "25":None, "26":None, "27":None, "28":None, "29":None, "30":None,
                                "31": None, "32":None, "33":None}
    elif dataset_name == 'OASI1_MRB':
        df_specific_dataset = {"GM": None, "WM": None, "CSF": None}
    elif dataset_name == 'fastMRI':
        df_specific_dataset = {"GM": None, "WM": None, "CSF": None}
    elif dataset_name == 'Prostate':
        df_specific_dataset = {"Seg1": None, "Seg2": None}

    df_last = {"shotname": None, "slice": None} 
    df = {**df_general, **df_specific_dataset, **df_last}
    return df

def get_rice_noise(img, snr=10, mu=0.0, sigma=1):
    
    level = snr * torch.max(img) / 3000
    
    
    
    
    size = img.shape
    x = level * torch.randn(size).to(device, dtype=torch.float) * sigma
    y = level * torch.randn(size).to(device, dtype=torch.float) * sigma
    x = x + img

    return torch.sqrt(x**2 + y**2)

if __name__ == "__main__":
    pass