import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
import os

#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


class SOLOHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 epsilon=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 mask_loss_cfg=dict(weight=3),
                 cate_loss_cfg=dict(gamma=2,
                                alpha=0.25,
                                weight=1),
                 postprocess_cfg=dict(cate_thresh=0.2,
                                      ins_thresh=0.5,
                                      pre_NMS_num=50,
                                      keep_instance=5,
                                      IoU_thresh=0.5)):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg
        # initialize the layers for cate and mask branch, and initialize the weights
        self._init_layers()
        self._init_weights()

        # check flag
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)
        pass

    # This function build network layer for cate and ins branch
    # it builds 4 self.var
        # self.cate_head is nn.ModuleList 7 inter-layers of conv2d
        # self.ins_head is nn.ModuleList 7 inter-layers of conv2d
        # self.cate_out is 1 out-layer of conv2d
        # self.ins_out_list is nn.ModuleList len(self.seg_num_grids) out-layers of conv2d, one for each fpn_feat
    def _init_layers(self):
        ## TODO initialize layers: stack intermediate layer and output layer
        # define groupnorm
        num_groups = 32

        self.layers_need_init = []

        # initial the category branch
        self.cate_head = nn.ModuleList()
        for i in range(7):
            conv = nn.Conv2d(256,256,kernel_size = 3,stride=1,padding=1,bias=False)
            self.layers_need_init.append(conv)
            seq = nn.Sequential(conv, nn.GroupNorm(num_groups,256),nn.ReLU())
            self.cate_head.append(seq)
        
        # intialize cate_out
        conv = nn.Conv2d(256, self.cate_out_channels, kernel_size = 3,padding=1,bias = True)
        self.layers_need_init.append(conv)
        seq = nn.Sequential(conv, nn.Sigmoid())
        self.cate_out = seq

        # Intialize mask branch
        self.ins_head = nn.ModuleList()
        conv = nn.Conv2d(258,256,kernel_size=3 , stride=1, padding=1, bias = False)
        self.layers_need_init.append(conv)
        seq = nn.Sequential(conv,nn.GroupNorm(num_groups,256),nn.ReLU())
        self.ins_head.append(seq)
        for i in range(6):
            conv = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias = False)
            self.layers_need_init.append(conv)
            seq = nn.Sequential(conv, nn.GroupNorm(num_groups,256),nn.ReLU())
            self.ins_head.append(seq)
        
        # Initialize mask out
        self.ins_out_list = nn.ModuleList()
        for i in range(len(self.seg_num_grids)):
            conv = nn.Conv2d(256, (self.seg_num_grids[i])**2, kernel_size=1, bias = True)
            self.layers_need_init.append(conv)
            seq = nn.Sequential(conv, nn.Sigmoid())
            self.ins_out_list.append(seq)



    # This function initialize weights for head network
    def _init_weights(self):
      # for layer in self.layers_need_init:
      #       nn.init.xavier_uniform_(layer.weight.data)
        for layer in self.cate_head:
          for m in layer:
            if m._class.name_.find('conv') !=-1:
              nn.init.xavier_uniform_(m.weight.data)
        # for layer in self.cate_out:
        #   for m in layer:
        #     if m._class.name_.find('conv') !=-1:
        #       nn.init.xavier_uniform_(m.weight.data)
        nn.init.xavier_uniform_(self.cate_out[0].weight.data)


        for layer in self.ins_head:
          for m in layer:
            if m._class.name_.find('conv') !=-1:
              nn.init.xavier_uniform_(m.weight.data)
        for layer in self.ins_out_list:
          for m in layer:
            if m._class.name_.find('conv') !=-1:
              nn.init.xavier_uniform_(m.weight.data)


    # Forward function should forward every levels in the FPN.
    # this is done by map function or for loop
    # Input:
        # fpn_feat_list: backout_list of resnet50-fpn
    # Output:
        # if eval = False
            # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred_list: list, len(fpn_level), each (bz,S,S,C-1) / after point_NMS
            # ins_pred_list: list, len(fpn_level), each (bz, S^2, Ori_H, Ori_W) / after upsampling
    def forward(self,
                fpn_feat_list,
                eval=False):
        new_fpn_list = self.NewFPN(fpn_feat_list)  # stride[8,8,16,32,32]
        assert new_fpn_list[0].shape[1:] == (256,100,136)
        quart_shape = [new_fpn_list[0].shape[-2]*2, new_fpn_list[0].shape[-1]*2]  # stride: 4

        # Generate the forward predictions for each feature pyramid level in parallel
        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level, 
            new_fpn_list,
            list(range(len(new_fpn_list))), 
            eval=eval,
            upsample_shape = quart_shape
        )

        assert len(new_fpn_list) == len(self.seg_num_grids)

        # assert cate_pred_list[1].shape[1] == self.cate_out_channels
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1]**2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]
        return cate_pred_list, ins_pred_list

    # This function upsample/downsample the fpn level for the network
    # In paper author change the original fpn level resolution
    # Input:
        # fpn_feat_list, list, len(FPN), stride[4,8,16,32,64]
    # Output:
    # new_fpn_list, list, len(FPN), stride[8,8,16,32,32]
    def NewFPN(self, fpn_feat_list):
        new_fpn_list = fpn_feat_list
        new_fpn_list[0] = torch.nn.functional.interpolate(new_fpn_list[0], size=(100, 136), mode = 'bilinear', align_corners=False)
        new_fpn_list[4] = torch.nn.functional.interpolate(new_fpn_list[4], size=(25, 34), mode = 'bilinear', align_corners=False)
        return new_fpn_list


    # This function forward a single level of fpn_featmap through the network
    # Input:
        # fpn_feat: (bz, fpn_channels(256), H_feat, W_feat)
        # idx: indicate the fpn level idx, num_grids idx, the ins_out_layer idx
    # Output:
        # if eval==False
            # cate_pred: (bz,C-1,S,S)
            # ins_pred: (bz, S^2, 2H_feat, 2W_feat)
        # if eval==True
            # cate_pred: (bz,S,S,C-1) / after point_NMS
            # ins_pred: (bz, S^2, Ori_H/4, Ori_W/4) / after upsampling
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## Notice, we distinguish the training and inference.
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cate_pred = fpn_feat
        ins_pred = fpn_feat # bz, 256, H_feat, W_feat
        num_grid = self.seg_num_grids[idx]  # current level grid

        # Create normalized coordinate channels
        H_feat = ins_pred.shape[2]
        W_feat = ins_pred.shape[3]
        h_coord_channel = (torch.ones((H_feat,W_feat), device=device) * torch.unsqueeze(torch.arange(H_feat, device = device), 1)) / (H_feat - 1)
        w_coord_channel = torch.ones((H_feat,W_feat), device=device) * torch.arange(W_feat, device = device) / (W_feat - 1)
        h_coord_channel = h_coord_channel*2 - 1 # H_feat, W_feat
        w_coord_channel = w_coord_channel*2 - 1 # H_feat, W_feat

        # Concatenate them to ins_pred
        h_coord_channel_extra = torch.unsqueeze(torch.unsqueeze(h_coord_channel, 0), 0).repeat(ins_pred.shape[0], 1, 1, 1) # 1,1,H_feat, W_feat
        w_coord_channel_extra = torch.unsqueeze(torch.unsqueeze(w_coord_channel, 0), 0).repeat(ins_pred.shape[0], 1, 1, 1) # 1,1,H_feat, W_feat
        ins_pred = torch.cat((ins_pred, h_coord_channel_extra), dim=1)
        ins_pred = torch.cat((ins_pred, w_coord_channel_extra ), dim=1)


        # Apply the layers of the mask branch of the network
        for layer in self.ins_head:
            ins_pred = layer(ins_pred)

        #####
        ##### Resizing mask branch output to 2H x 2W
        #####
        ins_pred = torch.nn.functional.interpolate(ins_pred, scale_factor=2, mode = 'bilinear', align_corners=False)

        # Apply the proper final layer for the current feature pyramid level
        ins_pred = self.ins_out_list[idx](ins_pred)


        # Align the fpn features before sending them into 
        # the category branch of the network
        cate_pred = F.interpolate(cate_pred, size = (num_grid, num_grid), mode = 'bilinear', align_corners=False)
        # Apply the layers of the category branch of the network
        for layer in self.cate_head:
            cate_pred = layer(cate_pred)
        # Apply the final layer for the category branch of the network
        cate_pred = self.cate_out(cate_pred)


        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ins_pred = F.interpolate(ins_pred, size = upsample_shape, mode = 'bilinear', align_corners=False)
            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred

    # Credit to SOLO Author's code
    # This function do a NMS on the heat map(cate_pred), grid-level
    # Input:
        # heat: (bz,C-1, S, S)
    # Output:
        # (bz,C-1, S, S)
    def points_nms(self, heat, kernel=2):
        # kernel must be 2
        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    # This function compute loss for a batch of images
    # input:
        # cate_pred_list: list, len(fpn_level), each (bz,C-1,S,S)
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    # output:
        # cate_loss, mask_loss, total_loss
    def loss(self,cate_pred_list,ins_pred_list,ins_gts_list,ins_ind_gts_list,cate_gts_list):
        ## TODO: compute loss, vecterize this part will help a lot. To avoid potential ill-conditioning, if necessary, add a very small number to denominator for focalloss and diceloss computation.
        ins_gts = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img,...]for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level)], 0)for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))]

        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img,...]for ins_preds_level_img, ins_ind_labels_level_img in zip(ins_preds_level, ins_ind_labels_level)], 0)for ins_preds_level, ins_ind_labels_level in zip(ins_pred_list, zip(*ins_ind_gts_list))]
        ## uniform the expression for cate_gts & cate_preds
        # cate_gts: (bz*fpn*S^2,), img, fpn, grids
        # cate_preds: (bz*fpn*S^2, C-1), ([img, fpn, grids], C-1)
        cate_gts = [torch.cat([cate_gts_level_img.type(torch.LongTensor).flatten()for cate_gts_level_img in cate_gts_level]) for cate_gts_level in zip(*cate_gts_list)]
        cate_gts = torch.cat(cate_gts)
        cate_preds = [cate_pred_level.permute(0,2,3,1).reshape(-1, self.cate_out_channels) for cate_pred_level in cate_pred_list]
        cate_preds = torch.cat(cate_preds,0)
        cate_loss = self.FocalLoss(cate_preds,cate_gts)
        mask_loss = self.DiceLoss(ins_preds,ins_gts)
        weighted_cate_loss = self.cate_loss_cfg['weight']*cate_loss
        weighted_mask_loss = self.mask_loss_cfg['weight']*mask_loss
        loss = weighted_cate_loss + weighted_mask_loss
        return weighted_cate_loss, weighted_mask_loss, loss

    def dMask(self,pred_mask,gtruth_mask):
        ans = 0
        num = torch.sum(pred_mask*gtruth_mask)
        den1 = torch.sum(pred_mask*pred_mask)
        den2 = torch.sum(gtruth_mask*gtruth_mask)
        ans = (2*num)/(den1+den2+0.00000001)
        return (1-ans)

    # This function compute the DiceLoss
    # Input:
        # mask_pred: (2H_feat, 2W_feat)
        # mask_gt: (2H_feat, 2W_feat)
    # Output: dice_loss, scalar
    def DiceLoss(self, mask_pred, mask_gt):
        ## TODO: compute DiceLoss
        tot = torch.zeros((1,))[0]+0.0000000001
        diceloss = 0
        for i,(pred,gt) in enumerate(zip(mask_pred,mask_gt)):
            for j,(pred_mask,gtruth_mask) in enumerate(zip(pred,gt)):
                diceloss += self.dMask(pred_mask,gtruth_mask)
            tot += pred.shape[0]
        return (diceloss/tot)

    # This function compute the cate loss
    # Input:
        # cate_preds: (num_entry, C-1)
        # cate_gts: (num_entry,)
    # Output: focal_loss, scalar
    def FocalLoss(self, cate_preds, cate_gts):
        ## TODO: compute focalloss
        cate_gts_one_hot = torch.zeros_like(cate_preds);
        for i,cl in enumerate(cate_gts):
            if cl.item()!=0:
                cate_gts_one_hot[i,cl.item()-1] = 1
        cate_gts_one_hot = torch.flatten(cate_gts_one_hot)
        p = torch.flatten(cate_preds)

        p_t = (cate_gts_one_hot*p)+((1-cate_gts_one_hot)*(1-p))
        a_t = (cate_gts_one_hot*self.cate_loss_cfg['alpha'])+ ((1-cate_gts_one_hot)*(1-self.cate_loss_cfg['alpha']))

        fl = (torch.sum(-a_t*(torch.pow((1-p_t),(self.cate_loss_cfg['gamma'])))*torch.log(p_t)))/(len(p_t))
        return fl

    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)

        return tuple(map(list, zip(*map_results)))

    # This function build the ground truth tensor for each batch in the training
    # Input:
        # ins_pred_list: list, len(fpn_level), each (bz, S^2, 2H_feat, 2W_feat)
        # / ins_pred_list is only used to record feature map
        # bbox_list: list, len(batch_size), each (n_object, 4) (x1y1x2y2 system)
        # label_list: list, len(batch_size), each (n_object, )
        # mask_list: list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # ins_gts_list: list, len(bz), list, len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), list, len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), list, len(fpn), (S, S), {1,2,3}
    def target(self,
               ins_pred_list,
               bbox_list,
               label_list,
               mask_list):

        # Generate ground truth image for each image in the batch in parallel
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.MultiApply(
            self.target_single_img,
            bbox_list,
            label_list,
            mask_list,
            featmap_sizes = [ins_pred.shape[-2:] for ins_pred in ins_pred_list]
        )

        # check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1]**2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1]**2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list
   

    def quantize_helper(self, value, total, number_of_grid_cells):
        return int(value / total * number_of_grid_cells)


    # -----------------------------------
    ## process single image in one batch
    # -----------------------------------
    # input:
        # gt_bboxes_raw: n_obj, 4 (x1y1x2y2 system)
        # gt_labels_raw: n_obj,
        # gt_masks_raw: n_obj, H_ori, W_ori
        # featmap_sizes: list of shapes of featmap
    # output:
        # ins_label_list: list, len: len(FPN), (S^2, 2H_feat, 2W_feat)
        # cate_label_list: list, len: len(FPN), (S, S)
        # ins_ind_label_list: list, len: len(FPN), (S^2, )
    def target_single_img(self,
                          gt_bboxes_raw,
                          gt_labels_raw,
                          gt_masks_raw,
                          featmap_sizes=None):
        ## TODO: finish single image target build
        # compute the area of every object in this single image

        # initial the output list, each entry for one featmap
        with torch.no_grad():
          ins_label_list = []
          ins_ind_label_list = []
          cate_label_list = []

          # For each feature pyramid level
          for fpn_idx, num_grid in enumerate(self.seg_num_grids): # for each FPN Level
              # Get the feature map height and width
              fpn_h = featmap_sizes[fpn_idx][0]
              fpn_w = featmap_sizes[fpn_idx][1]


              # Get the instance scale range for this feature pyramid level
              fpn_scale_min = self.scale_ranges[fpn_idx][0]
              fpn_scale_max = self.scale_ranges[fpn_idx][1]

              # Initialize ground truth tensors for the feature pyramid level
              device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
              ins_label = torch.zeros((num_grid ** 2, fpn_h, fpn_w), device = device)
              tensor1 = torch.randn((num_grid**2,),device = device)
              ins_ind_label =  torch.randn(tensor1.size()).bool()
              ins_ind_label[:] = False
              # ins_ind_label = torch.zeros((num_grid ** 2,), device = device)
              cate_label = torch.zeros((num_grid, num_grid), device = device)

              # For each object
              for obj_idx in range(len(gt_bboxes_raw)):
                  # Get the information about the object
                  x1, y1, x2, y2 = gt_bboxes_raw[obj_idx]
                  # print("bounding box vaiables",x1,y1,x2,y2)
                  gt_label = gt_labels_raw[obj_idx]
                  gt_mask = gt_masks_raw[obj_idx]

                  # Calculate the objects scale
                  obj_scale = ((x2-x1) *  (y2-y1)) ** 0.5

                  if obj_scale >= fpn_scale_min and obj_scale < fpn_scale_max:
                      # The object's scale fits into the scale range for this feature pyramid level

                      # Calculate object's center
                      center_h, center_w = ndimage.measurements.center_of_mass(gt_mask.data.cpu().numpy())
                      

                      # Calculate the object's center region in terms of grid cells
                      ori_h, ori_w = gt_mask.shape[0], gt_mask.shape[1]
                      half_h, half_w = (self.epsilon * (y2-y1)) / 2.0,  (self.epsilon * (x2 - x1)) / 2.0
                      # print("half values :", half_h, half_w)
                      center_grid_h = self.quantize_helper(center_h, ori_h, num_grid)
                      center_grid_w = self.quantize_helper(center_w, ori_w, num_grid)
                      top_ind = max(0, self.quantize_helper(center_h - half_h, ori_h, num_grid))
                      bottom_ind = min(num_grid - 1, self.quantize_helper(center_h + half_h, ori_h, num_grid))
                      left_ind = max(0, self.quantize_helper(center_w - half_w, ori_w, num_grid))
                      right_ind = min(num_grid - 1, self.quantize_helper(center_w + half_w, ori_w, num_grid))

                      # print("center_grid_h",center_grid_h,"center_grid_w",center_grid_w)
                      # print("top ind :",top_ind,"bottom_ind : ", bottom_ind)
                      # print("left ind :",left_ind,"right_ind : ", right_ind)

                      # Constrain the center region to be a maximum of 3 x 3 grid cells
                      top = max(top_ind, center_grid_h - 1)
                      bottom = min(bottom_ind, center_grid_h + 1)
                      left = max(left_ind, center_grid_w - 1)
                      right = min(right_ind, center_grid_w + 1)

                      # print("top :",top,"bottom :",bottom)
                      # print("left :",left,"right :",right)

                      # Scale the ground truth mask
                      gt_mask_scaled = F.interpolate(torch.unsqueeze(torch.unsqueeze(gt_mask, 0), 0), size = (fpn_h, fpn_w),mode = 'bilinear', align_corners=False)[0][0]
                      
                      # Iterate over center region's grid cells and set/active 
                      # the appropriate ground truth tensors values for each
                      # grid cell
                      
                      for i in range(top, bottom + 1):
                          for j in range(left, right + 1):
                            
                            ij = i * num_grid + j # Flat grid cell index in S^2
                            ins_label[ij] = gt_mask_scaled
                            ins_ind_label[ij] = True
                            cate_label[i][j] = gt_label


              # Finish the ground truth tensor for the feature pyramid level,
              # add it to the results
              ins_label_list.append(ins_label)
              ins_ind_label_list.append(ins_ind_label)
              cate_label_list.append(cate_label)

        # check flag
        assert ins_label_list[1].shape == (1296,200,272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)
        return ins_label_list, ins_ind_label_list, cate_label_list

    # This function receive pred list from forward and post-process
    # Input:
        # ins_pred_list: list, len(fpn), (bz,S^2,Ori_H/4, Ori_W/4)
        # cate_pred_list: list, len(fpn), (bz,S,S,C-1)
        # ori_size: [ori_H, ori_W]
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)

    def PostProcess(self,ins_pred_list,cate_pred_list,ori_size):
        ## TODO: finish PostProcess
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ins_pred_img  = torch.Tensor(0).to(device)
        cate_pred_img = torch.Tensor(0).to(device)
        NMS_sorted_scores_list = []
        NMS_sorted_cate_label_list = []
        NMS_sorted_ins_list = []

        for i in range(len(ins_pred_list[0])):
            for j in range(len(ins_pred_list)):
                cate_pred = cate_pred_list[j][i].reshape(-1,3) 
                ins_pred_img = torch.cat((ins_pred_img,ins_pred_list[j][i]),0)

                cate_pred_img = torch.cat((cate_pred_img,cate_pred),0)
                
            NMS_sorted_scores,NMS_sorted_ins,NMS_sorted_cate_label= self.PostProcessImg(ins_pred_img,cate_pred_img,ori_size)
            NMS_sorted_scores_list.append(NMS_sorted_scores)
            NMS_sorted_cate_label_list.append(NMS_sorted_cate_label)
            NMS_sorted_ins_list.append(NMS_sorted_ins)

        
        return NMS_sorted_scores_list,NMS_sorted_cate_label_list,NMS_sorted_ins_list


    # This function Postprocess on single img
    # Input:
        # ins_pred_img: (all_level_S^2, ori_H/4, ori_W/4)
        # cate_pred_img: (all_level_S^2, C-1)
    # Output:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
    def PostProcessImg(self,ins_pred_img,cate_pred_img,ori_size):
        cate_pred_img_scores,cate_pred_img_class = torch.max(cate_pred_img,dim=1)
        cate_pred_img_ind = (cate_pred_img_scores[:]>0.2).nonzero()
        new_cate_pred_img_scores = cate_pred_img_scores[cate_pred_img_ind[:]]
        new_cate_pred_img_class = cate_pred_img_class[cate_pred_img_ind[:]]
        new_ins_pred_img = torch.squeeze(ins_pred_img[cate_pred_img_ind[:]])
        scores = []
        for i in range(len(new_cate_pred_img_scores)):
            num = 0
            tot = 0.000000000001
            for j in range(len(new_ins_pred_img[i])):
                # print(new_ins_pred_img[i].shape)
                for k in range(len(new_ins_pred_img[i][j])):
                    # print(new_ins_pred_img[i][j][k])
                    if new_ins_pred_img[i][j][k] > 0.5:
                        num += new_ins_pred_img[i][j][k]
                        tot +=1
            scores.append(num*new_cate_pred_img_scores[i]/tot)
        # sorted_ins = [y for _,y in sorted(zip(scores,new_ins_pred_img))
        x,y,z = zip(*[(x,y,z) for x,y,z in sorted(zip(scores,new_ins_pred_img,new_cate_pred_img_class))])
        sorted_scores = torch.stack(x)
        sorted_new_ins_pred_img = torch.stack(y)
        sorted_new_cate_pred_img = torch.stack(z)
        # sorted_scores = sort_together([scores,new_ins_pred_img,new_cate_pred_img])[0]
        # sorted_new_ins_pred_img = sort_together([scores,new_ins_pred_img,new_cate_pred_img])[1]
        # sorted_new_cate_pred_img = sort_together([scores,new_ins_pred_img,new_cate_pred_img])[2]

        new_NMS_scores = self.MatrixNMS(sorted_new_ins_pred_img,sorted_scores)
        # print("NMS",new_NMS_scores[0], type(new_NMS_scores),len(new_NMS_scores))
        # new_sorted_ins = [y for ,y in sorted(zip(new_NMS_scores,sorted_ins))]
        try:
            x,y,z = zip(*[(x,y,z) for x,y,z in sorted(zip(new_NMS_scores,sorted_new_ins_pred_img,sorted_new_cate_pred_img))])
        except:
            return [[]], [[]], [[]] 
        new_NMS_scores = (torch.from_numpy(np.asarray(x)))
        # print("NMS score shape",new_NMS_scores.shape)
        sorted_new_ins_pred_img = torch.stack(y)
        sorted_new_cate_pred_img = torch.stack(z)
        if len(new_NMS_scores)<=5:
          
            sorted_new_ins_pred_img = F.interpolate(torch.unsqueeze(sorted_new_ins_pred_img,0), size = ori_size)[0]
            return new_NMS_scores,sorted_new_ins_pred_img,sorted_new_cate_pred_img
        else:
            # print(sorted_new_ins_pred_img.shape)
            clipped_new_NMS_scores = new_NMS_scores[-5:]
            sorted_new_ins_pred_img = F.interpolate(torch.unsqueeze(sorted_new_ins_pred_img,0), size = ori_size)[0]
            clipped_ins_pred_img = sorted_new_ins_pred_img[-5:]
            clipped_cate_pred_img = sorted_new_cate_pred_img[-5:]
            return clipped_new_NMS_scores,clipped_ins_pred_img,clipped_cate_pred_img

    # This function perform matrix NMS
    # Input:
        # sorted_ins: (n_act, ori_H/4, ori_W/4)
        # sorted_scores: (n_act,)
    # Output:
        # decay_scores: (n_act,)
    def MatrixNMS(self, sorted_ins, sorted_scores, method='gauss', gauss_sigma=0.5):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # reshape to N x HW
        masks = torch.reshape(sorted_ins, (len(sorted_ins), len(sorted_ins[0]) * len(sorted_ins[0][0])))

        # precompute the IoU matrix: NxN
        intersection = masks @ torch.transpose(masks, 0, 1)
        areas = torch.sum(intersection, dim=1).repeat(sorted_ins.shape[0], 1)
        union = areas + torch.transpose(areas, 0, 1) - intersection
        ious = torch.triu(intersection / union, diagonal = 1)

        # max IoU for each: NxN
        ious_cmax = torch.max(ious, dim = 0).values
        ious_cmax = torch.transpose(ious_cmax.repeat(sorted_ins.shape[0], 1), 0, 1)

        # MatrixNMS: NxN
        if method == 'gauss':
            decay = torch.exp(-1 * (ious * 2 - ious_cmax * 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)

        # decay factor: N
        decay = decay.min(dim=0).values
        # print("decay:",decay)
        # print("sorted_scores:" , sorted_scores, len(sorted_scores))
        # print("decay array",decay,decay.shape)
        # print((sorted_scores * torch.unsqueeze(decay,dim=1)).shape)
        p = sorted_scores * torch.unsqueeze(decay,dim=1)
        return p.cpu().data.numpy().flatten().tolist()

    # -----------------------------------
    ## The following code is for visualization
    # -----------------------------------
    # this function visualize the ground truth tensor
    # Input:
        # ins_gts_list: list, len(bz), len(fpn), (S^2, 2H_f, 2W_f)
        # ins_ind_gts_list: list, len(bz), len(fpn), (S^2,)
        # cate_gts_list: list, len(bz), len(fpn), (S, S), {1,2,3}
        # color_list: list, len(C-1)
        # img: (bz,3,Ori_H, Ori_W)
        ## self.strides: [8,8,16,32,32]
    def PlotGT(self,
               ins_gts_list,
               ins_ind_gts_list,
               cate_gts_list,
               color_list,
               img):

        # Loop over each image in the batch
        for cur in range(len(ins_gts_list)):
            # Get the data for the current image in the batch
            cur_ins = ins_gts_list[cur]
            cur_ins_ind = ins_ind_gts_list[cur]
            cur_cate = cate_gts_list[cur]
            cur_img = img[cur]

            # Get the number of feature pyramid levels
            fpn_levels = len(cur_ins)

            # Plot each feature pyramid level for the image
            for fpn_idx, num_grid in enumerate(self.seg_num_grids): # for each FPN Level 
                cur_img_normalized = transforms.functional.normalize(cur_img, (-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
                cur_img_permuted = cur_img_normalized.permute(1,2,0).type(torch.long)
                plt.imshow(cur_img_permuted.data.numpy())
                plt.title("Image "+str(cur)+" in Batch -- Feature Pyramid Level " + str(fpn_idx))

                for ij in torch.nonzero(cur_ins_ind[fpn_idx].float(), as_tuple = True)[0]:
                    ij = ij.item()
                    i = int(ij / num_grid)
                    j = ij - i * num_grid
                    mask = cur_ins[fpn_idx][ij]
                    category = cur_cate[fpn_idx][i][j]
                    color = color_list[int(category.item()) - 1]

                    # Scale up the mask to the image size
                    mask = F.interpolate(torch.unsqueeze(torch.unsqueeze(mask, 0), 0), size = (cur_img_permuted.shape[0], cur_img_permuted.shape[1]), mode = 'bilinear', align_corners=False)[0][0]

                    # Show the mask
                    mask_new = np.ma.masked_where(mask.data.numpy() == 0, mask.data.numpy())
                    plt.imshow(mask_new, cmap=color, alpha=0.25)
                    found_mask = True

                plt.show()

    # This function plot the inference segmentation in img
    # Input:
        # NMS_sorted_scores_list, list, len(bz), (keep_instance,)
        # NMS_sorted_cate_label_list, list, len(bz), (keep_instance,)
        # NMS_sorted_ins_list, list, len(bz), (keep_instance, ori_H, ori_W)
        # color_list: ["jet", "ocean", "Spectral"]
        # img: (bz, 3, ori_H, ori_W)
    def PlotInfer(self,
                  NMS_sorted_scores_list,
                  NMS_sorted_cate_label_list,
                  NMS_sorted_ins_list,
                  color_list,
                  img,
                  iter_ind):

        # Loop over each image in the batch
        for cur in range(len(img)):
            # Get the data for the current image in the batch
            cur_ins = NMS_sorted_ins_list[cur]
            cur_scores = NMS_sorted_scores_list[cur]
            cur_cate = NMS_sorted_cate_label_list[cur]
            # print("category :",cur_cate)
            # print("scores :", cur_scores)

            cur_img = img[cur]


            cur_img_normalized = transforms.functional.normalize(cur_img, (-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
            cur_img_permuted = cur_img_normalized.permute(1,2,0).type(torch.long)
            plt.imshow(cur_img_permuted.cpu().data.numpy())
            plt.title("Image "+str(cur)+" in Batch ")


            # Loop over each keep instance
            for keepi in range(len(cur_ins)):
                score = cur_scores[keepi]
                # If the score is above a threshold display the mask
                if score > 0:

                    mask = cur_ins[keepi]
                    # print("mask shape : ",mask.shape)
                    category = cur_cate[keepi]
                    category = min(max(round(int(category.item())), 0), 3)
                    color = color_list[category]

                    # Make a "hard mask" by thresholding
                    # < .5 instead of > .5 since we are doing mask instead of 1-mask
                    # mask = mask.data.numpy()
                    # threshold_mask = (mask > .5).astype(np.uint8)
                    # mask = threshold_mask * mask

                    # Show the mask
                    # mask_new = np.ma.masked_where(mask < 0.5, mask)
                    
                    # found_mask = True
                    # Make a "hard mask" by thresholding
                    # < .5 instead of > .5 since we are doing mask instead of 1-mask
                    mask = mask.cpu().data.numpy()
                    threshold_mask = (mask < .5).astype(np.uint8)

                    # Show the mask
                    mask_new = np.ma.masked_where(threshold_mask == 1, threshold_mask)
                    plt.imshow(mask_new, cmap=color, alpha=0.2)

            plt.show()
            plt.close()

from backbone import *
if __name__ == '__main__':

    if IN_COLAB:
        imgs_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_img_comp_zlib.h5")
        masks_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_mask_comp_zlib.h5")
        labels_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_labels_comp_zlib.npy")
        bboxes_path = os.path.join(HOMEWORK_FOLDER, "data/hw3_mycocodata_bboxes_comp_zlib.npy")
    else:
        imgs_path = os.path.join('.', "data/hw3_mycocodata_img_comp_zlib.h5")
        masks_path = os.path.join('.', "data/hw3_mycocodata_mask_comp_zlib.h5")
        labels_path = os.path.join('.', "data/hw3_mycocodata_labels_comp_zlib.npy")
        bboxes_path = os.path.join('.', "data/hw3_mycocodata_bboxes_comp_zlib.npy")

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target


        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list,ins_ind_gts_list,cate_gts_list,mask_color_list,img)


