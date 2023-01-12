
from torch import nn

class LocalStructuralLoss(nn.Module): 
    def __init__(self):
        super(LocalStructuralLoss, self).__init__()
        print('using local structural loss')
        self.weight = [] 
        self.loss_l1 = nn.SmoothL1Loss()
        
    def forward(self, rec_result, gt, seg_result, seg_lbl):
        B, C, H, W = seg_lbl.shape
        total_loss = 0
        for i in range(0, C):
            gt_local = gt * seg_lbl[:, i:i+1, ...]
            rec_local = rec_result * seg_lbl[:, i:i+1, ...]
            
            total_loss += self.loss_l1(rec_local, gt_local) 
        return total_loss

class FeatureLoss(nn.Module): 
    def __init__(self):
        super(FeatureLoss, self).__init__()
        print('using local structural loss')
        self.weight = [] 
        self.loss_l1 = nn.SmoothL1Loss()
        
    def forward(self, features_u, features_f):
        total_loss = 0
        for feat_u, feat_f in zip(features_u, features_f):
            
            total_loss += self.loss_l1(feat_u, feat_f) 
        return total_loss

class criterion():

    @staticmethod
    def recon_criterion(self, out_recon, full_img, seg_lable):
        rec_loss = self.criterion(out_recon, full_img)
        perceptual_loss = self.local_loss(out_recon, full_img, seg_lable, seg_lable) 
        loss = rec_loss + perceptual_loss * self.weight_local
        return loss
    @staticmethod
    def recon_criterion_with_perceptual_loss(self, out_recon, full_img, seg_lable):
        rec_loss = self.criterion(out_recon, full_img)
        perceptual_loss = self.perceptual_loss(out_recon, full_img) 
        loss = rec_loss + perceptual_loss['prc'] * self.weight_perceptual_prc + perceptual_loss['style'] * self.weight_perceptual_style
        return loss
    @staticmethod
    def recon_criterion_only_perceptual_loss(self, out_recon, full_img, seg_lable):
        
        perceptual_loss = self.perceptual_loss(out_recon, full_img) 
        loss = perceptual_loss['prc'] * 1 + perceptual_loss['style'] * 10
        return loss
    @staticmethod
    def comb_valina_criterion(self, out_recon, full_img, out_seg, seg_lable): 
        loss_recon = self.criterion[0](out_recon, full_img)
        loss_seg = self.criterion[1](out_seg, seg_lable)
        
        loss = self.weight_recon * loss_recon + self.weight_seg * loss_seg
        return loss
    @staticmethod
    def comb_criterion(self, out_recon, full_img, out_seg, seg_lable, edge): 
        loss_recon = self.criterion[0](out_recon, full_img)
        loss_seg = self.criterion[1](out_seg, seg_lable)
        loss_local_recon = self.criterion[2](out_recon, full_img, edge, edge)
        loss = self.weight_recon * loss_recon + self.weight_seg * loss_seg + self.weight_local * loss_local_recon
        return loss
    @staticmethod
    def comb_criterion_perceptual(self, out_recon, full_img, out_seg, seg_lable): 
        loss_recon = self.criterion[0](out_recon, full_img)
        loss_seg = self.criterion[1](out_seg, seg_lable)
        
        perceptual_loss = self.perceptual_loss(out_recon, full_img) 
        loss = self.weight_recon * loss_recon + self.weight_seg * loss_seg + \
             perceptual_loss['prc'] * self.weight_perceptual_prc + perceptual_loss['style'] * self.weight_perceptual_style
        return loss