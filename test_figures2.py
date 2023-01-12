import sys
import torch
import warnings


import numpy as np

import argparse
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.append('.') # 
from data.dataset import get_h5py_mixed_test_dataset
from runner.csmri1 import ModelSet as CSMRI1ModelSet
from testing.test_toolbox import Dual_model, createinput

from runner.csmtl import ModelSet as CSMTLModelSet
from runner.csl_seq_unet import ModelSet as CSLModelSet
from runner.main.asl_mixed_ablation_seq_mdrec_v2_step3_1_bg_localloss import ModelSet as ASLModelSet
from runner.pretrained_seg import SegModelSet

from utils.dualdomain_tool import create_complex_value
from utils.train_utils import return_data_ncl_imgsize
from testing.test_toolbox import tensor2np, seg2np, get_metrics, get_seg_oasis1_main_map,  dataframe_template, get_masks_map_onebyone

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class NameSpace(object):
    def __init__(self, figs):

        if figs == 2:
            self.bbox1 = (128, 100, 128+60, 100+30)
            self.bbox2 = (95, 170, 95+60, 170+30) 
            self.THE_ONE = 1 # random choosing for testing     

#######################################-TEST-#########################################################

def test():

    data_name = args.dataset_name
    # traj_type = args.mask
    desired_sparsity = args.rate
    args.desired_sparsity = args.rate
    args.nclasses, inputs_size = return_data_ncl_imgsize(args.dataset_name)
    # args.nclasses = 1
    if args.maskType == '1D':
        traj_type = 'cartesian'
    elif args.maskType == '2D':
        traj_type = 'random'

    args.mask = traj_type

    namespace = NameSpace(args.figs)

    acc = int(desired_sparsity * 100)

    val_loader = get_h5py_mixed_test_dataset(args.dataset_name)
    
    
    dataloader = val_loader
    # model csmri1
    modelset = CSMRI1ModelSet(args, test=True)
    baselinemodel = modelset.model  
    baselinemodel.eval()

    # model csmri2
    create_dual_model = Dual_model(args, acc=acc, mask_type = traj_type )
    dualmodel = create_dual_model.model.to(device, dtype=torch.float)
    mask_torch = create_dual_model.mask_torch.to(device, dtype=torch.float)
    dualmodel.eval()

    # model csmtl
    args.stage = '1'
    modelset = CSMTLModelSet(args, test=True)
    liumodel = modelset.model     
    liumodel.eval()

    # mode = csl
    args.stage = '1'
    modelset = CSLModelSet(args, test=True)
    recon_model = modelset.model
    recon_model.eval()
    modelset = SegModelSet(args, test=True)
    seg_model = modelset.model
    seg_model.eval()

    # # combine_model: mtl1
    args.stage = '1'
    modelset = ASLModelSet(args, test=True)
    comb_model = modelset.model
    comb_model.eval()

    imgs = {}
    masks = {}
    segs = {}
    df = {'csmri1': None, 'csmri2': None, 'csmtl': None, 'csl': None, 'asl':None, 'supre':None}

    template = dataframe_template(dataset_name=data_name)
    # initialize
    for key in df.keys():
        df[key] = pd.DataFrame(template)

    NUM_SLICE = -1

    args.PREFIX = './demo_results/'
    with torch.no_grad():
        for x, y, y_, shotname in dataloader:
            
            inputs = x.to(device, dtype=torch.float)

            shotname = shotname[0]  
            THE_ONE =  namespace.THE_ONE # 11, 214， 111

            NUM_SLICE += 1
            if NUM_SLICE != THE_ONE:  # a specific shotname figure.
                continue
            seg_index = y.shape[1]-1

            y = y[:, 0:seg_index, ...]

            imgs['gt'] = tensor2np(x)
            segs['gt'] = seg2np(y)

            # model-baseline  (end-to-end for recon and seg)
            # recon_baseline, seg_baseline, uifft, complex_abs, mask, fft, undersample = baselinemodel(inputs)
            recon_baseline, uifft, complex_abs, mask, fft, undersample = baselinemodel(inputs)
            imgs['csmri1'] = tensor2np(recon_baseline)

            seg_baseline = seg_model(recon_baseline)  # upper bound of segmentation task from 100% sampling.
            segs['csmri1'] = seg2np(torch.sigmoid(seg_baseline))
            # segs['baseline'] = torch.sigmoid(seg_baseline).cpu().detach().numpy()
            masks['csmri1'] = tensor2np(mask) 

            # model-liu  (end-to-end for mask learning, recon and seg, but first guided by seg)
            recon_liu, seg_liu, uifft, complex_abs, mask, fft, undersample = liumodel(inputs)
            pred_results = liumodel(inputs)


            seg_liu = seg_liu[:, 0:seg_index,...]

            imgs['csmtl'] = tensor2np(recon_liu)
            segs['csmtl'] = seg2np(torch.sigmoid(seg_liu))

            # model-dual-net + seg-hrnet (two nets for recon and seg)
            u_img, u_k, img, k = createinput(inputs, mask_torch)
            recon_dual = dualmodel(*(u_k, u_img))
            seg_dual = seg_model(torch.sqrt(recon_dual[:, 0:1, ...]**2 + recon_dual[:, 1:2, ...]**2))
            recon_dual = create_complex_value(recon_dual[0].detach().cpu().numpy())
            recon_dual = np.abs(tensor2np(recon_dual))
            imgs['csmri2'] = recon_dual
            segs['csmri2'] = seg2np(torch.sigmoid(seg_dual))  # brow model-liu.

            # model-seq (for recon)
            # model-seq (for recon)
            pred_results = recon_model(inputs)
            out_recon, new_mask, zero_recon = (pred_results['output'], pred_results['mask'], pred_results['zero_recon'].detach())
            recon_loupe=out_recon
            imgs['csl'] = tensor2np(recon_loupe)


            masks['csl'] = pred_results['masks']
            # model_seg after loupe
            seg_loupe = seg_model(recon_loupe)
            segs['csl'] = seg2np(torch.sigmoid(seg_loupe))

            # model-ours  (end-to-end for mask learning, recon and seg)
            pred_results = comb_model(inputs)
            ours_crec, new_mask, zero_recon, ours_out_seg, ours_frec = (pred_results['output'], pred_results['mask'], 
                                                                    pred_results['zero_recon'].detach(),
                                                                    pred_results['out_seg'], pred_results['second_rec'])
            recon_ours = torch.norm(ours_frec, dim=1, keepdim=True)
            seg_ours = ours_out_seg[:, 0:seg_index,...]
            # seg_ours  = seg_ours[:, 0:3,...]

            imgs['asl'] = tensor2np(recon_ours)
            # masks['asl'] = tensor2np(new_mask)
            masks['asl'] = pred_results['masks']
            segs['asl'] = seg2np(torch.sigmoid(seg_ours))

            seg_supre = seg_model(inputs)  # upper bound of segmentation task from 100% sampling.
            segs['supre'] = seg2np(torch.sigmoid(seg_supre))

            get_metrics(args, imgs, segs, df, traj_type, shotname, NUM_SLICE, desired_sparsity)
            get_seg_oasis1_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace)
            get_masks_map_onebyone(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace) 


            for dfkey in df.keys():
                if dfkey in ['shotname', 'slice']:
                    continue
                else:
                    print('{}:'.format(dfkey))
                    print(df[dfkey].mean())
            
            break
#-----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=int, default=1, choices=[0,1], help="The learning rate") # 
    parser.add_argument("--lr", type=float, default=5e-4, help="The learning rate") # 5e-4, 5e-5, 5e-6
    parser.add_argument(
        "--batch_size", type=int, default=12, help="The batch size")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        choices=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25],  # cartesian_0.1 is bad;
        help="The undersampling rate")
   
    parser.add_argument(
        "--mask",
        type=str,
        default="random",
        choices=["cartesian", "radial", "random"],
        help="The type of mask")
              
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="The GPU device")
    parser.add_argument(
        "--bn",
        type=bool,
        default=False,
        choices=[True, False],
        help="Is there the batchnormalize")
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="which model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Prostate",
        help="which dataset")
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        choices=[True],
        help="If this program is a test program.")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="2D",
        choices=['1D', '2D'])
    parser.add_argument(
        "--nclasses",
        type=int,
        default=None,
        choices=[0,1,2,3,4,5,6,7,8],
        help="nclasses of segmentation")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--PREFIX",
        type=str,
        default="./testing/")
    parser.add_argument(
        "--stage",
        type=str,
        default='3',
        choices=['1','2', '3'],
        help="1: coarse rec net; 2: fixed coarse then train seg; 3: fixed coarse and seg, then train fine; 4: all finetune.")
    parser.add_argument(
        "--stage1_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage2_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage3_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--load_model",
        type=int,
        default=1,
        choices=[0,1],
        help="reload last parameter?")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None)
    parser.add_argument(
        "--maskType",
        type=str,
        default='2D',
        choices=['1D', '2D'])
    parser.add_argument(
        "--shotname",
        type=str,
        default=None)
    parser.add_argument(
        "--figs",
        type=int,
        default=None)

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    test()