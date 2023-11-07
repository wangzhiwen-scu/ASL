# FSL

This is the official PyTorch implementation of our manuscipt:

> [**Promoting fast MR imaging pipeline by full-stack AI**](xxx)       
> Zhiwen Wang, Bowen Li, Hui Yu, Zhongzhou Zhang, Maosong Ran, Wenjun Xia, Ziyuan Yang, Jingfeng Lu, Hu Chen, Jinfeng Lu, Jiliu Zhou, Hongming Shan, Yi Zhang        
> *Accepted by iScience*

## Getting started

###  1. Clone the repository
```bash
git clone https://github.com/wangzhiwen-scu/FSL.git
cd fsl
```


### 2. Install dependencies

Here's a summary of the key dependencies.
- python 3.7
- pytorch 1.7.1

We recommend using [conda](https://docs.conda.io/en/latest/) to install all of the dependencies.

```bash
conda env create -f environment.yaml
```
To activate the environment, run:

```bash
conda activate fsl
```

### 3. Pre-trained Model and Testing Dataset
All data and models can be downloaded in [Google-drive](https://drive.google.com/file/d/1fdxsNnbEURpetsH9seP4RRv9nML2y2i1/view?usp=sharing).

It is a `zip file` (~843M) which contain a `demo testing data` and `parameter files of compared models`. 

### 4. File Organization
Then place the `demo testing data` in:

```
├── datasets
│   ├── brain
│   │   ├── OASI1_MRB
│   │   ├── testing-h5py
│   │   │   ├── demo
│   │   │   │   └── oasis1_disc1_OAS1_0042_MR1.h5
│   ├── cardiac
│   └── prostate
```

place the `parameter files` in:
```
├── model_zoo
│   ├── pretrained_seg
│   │   └── OASI1_MRB_3seg.pth
│   └── tab1
│       └── OASI1_MRB
│           ├── asl_ablation_seqmdrecnet_bg_step3_1_local__0.05_2D.pth
│           ├── csl_seqmri_unet__0.05_2D.pth
│           ├── csmri1__0.05.pth
│           ├── csmri2__5.pth
│           └── csmtl__0.05.pth
```
### 5. Training

Please see [runner/main/asl_mixed_ablation_seq_mdrec_v2_step3_1_bg_localloss.py](runner/main/asl_mixed_ablation_seq_mdrec_v2_step3_1_bg_localloss.py) for an example of how to train FSL.


### 6. Testing

```
bash demo.sh
```

## Acknowledgement

Part of the subsampling learning network are adapted from **LOUPE** and **SeqMRI**. 
Part of the reconstruction network structures are adapted from **MD-Recon-Net**.
 
+ LOUPE: [https://github.com/cagladbahadir/LOUPE](https://github.com/cagladbahadir/LOUPE).
+ SeqMRI: [https://github.com/tianweiy/SeqMRI](https://github.com/tianweiy/SeqMRI).
+ MD-Recon-Net: [https://github.com/Deep-Imaging-Group/MD-Recon-Net](https://github.com/Deep-Imaging-Group/MD-Recon-Net).

Thanks a lot for their great works!

## contact
If you have any questions, please feel free to contact Wang Zhiwen {wangzhiwen_scu@163.com}.

<!-- ## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{st++,
  title={ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={xxx},
  year={xxx}
}

@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
``` -->