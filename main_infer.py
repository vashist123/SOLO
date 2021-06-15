import torch

# Set seed
torch.manual_seed(50)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from dataset import *
from functools import partial
from backbone import *
from solo_head import *
from main_train import *
from sklearn.metrics import auc
import os
import time
import itertools


#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

###########################################
# P/R Curve function
###########################################

def calculate_pr_info(total_positives, running_pr_info, ins_pred_list, ins_gts_list, cate_pred_list, cate_gts_list):
  # Everything is indexed by 0 since we are doing batch_size = 1 in infer mode
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  iou_threshold = .1
  flat_list_of_categories = list(itertools.chain.from_iterable([tensor.cpu().data.numpy().flatten().tolist() for tensor in cate_gts_list[0]]))
  total_positives[0] += sum([category == 1 for category in flat_list_of_categories])
  total_positives[1] += sum([category == 2 for category in flat_list_of_categories])
  total_positives[2] += sum([category == 3 for category in flat_list_of_categories])
  for fpn_index in range(len(cate_gts_list[0])):
      for i in range(cate_gts_list[0][fpn_index].shape[0]):
        for j in range(cate_gts_list[0][fpn_index].shape[1]):
            num_grid = cate_gts_list[0][fpn_index].shape[0]
            ij = i * num_grid + j # Flat grid cell index in S^2
            category_gt = cate_gts_list[0][fpn_index][i][j].item()
            category_prediction_scores = cate_pred_list[fpn_index][0][i][j].cpu().data.numpy().flatten()
            category_pred = np.argmax(category_prediction_scores) + 1
            predicted_correct = category_gt == category_pred
            mask_gts = ins_gts_list[0][fpn_index][ij].cpu().data.numpy()
            mask_pred = ins_pred_list[fpn_index][0][ij].cpu().data.numpy()
            mask_pred_thresholded_hard = (mask_pred > .5).astype(np.uint8)
            mask_pred_thresholded_soft = mask_pred_thresholded_hard * mask_pred  
            if category_gt > 0:
                confidence_score = category_pred * (np.sum(mask_pred_thresholded_soft) / (0.000000000001 + np.sum(mask_pred_thresholded_hard)))
                masks = torch.from_numpy(np.concatenate([mask_pred, mask_gts]).reshape((2, mask_gts.shape[0] * mask_gts.shape[1]))).to(device)    
                intersection = masks @ torch.transpose(masks, 0, 1)    
                areas = torch.sum(intersection, dim=1).repeat(2, 1)    
                union = areas + torch.transpose(areas, 0, 1) - intersection
                ious = torch.triu(intersection / union, diagonal = 1)
                ious_cmax = torch.max(ious, dim = 0).values
                iou = torch.max(ious_cmax, dim =0).values.item()
                predicted_correct = (iou > iou_threshold) and (category_gt == category_pred)
                running_pr_info[0].append((confidence_score, (category_gt == 1) and predicted_correct))
                running_pr_info[1].append((confidence_score, (category_gt == 2) and predicted_correct))
                running_pr_info[2].append((confidence_score, (category_gt == 3) and predicted_correct))


def average_precision(cls, total_positives, running_pr_info, return_curves = False):
  running_pr_info[cls].sort(reverse=True)
  
  correct = 0
  precisions = []
  recalls = []
  total_positives = [cls_total_positives.item() for cls_total_positives in total_positives]

  for i in range(len(running_pr_info[cls])):
    if running_pr_info[cls][i][1]:
      correct += 1
    precisions.append(correct / (i+1))
    if total_positives[cls] == 0:
        recalls.append(0)
    else:   
        recalls.append(correct / total_positives[cls])

  # print("Precisions:", precisions[0:500])
  # print("Recalls:", recalls[0:500])

  if not return_curves:
    return auc(recalls, precisions)
  else:
    return precisions, recalls, auc(recalls, precisions)

def mean_average_precision(total_positives, running_pr_info):
  ap0 = average_precision(0, total_positives, running_pr_info)
  ap1 = average_precision(1, total_positives, running_pr_info)
  ap2 = average_precision(2, total_positives, running_pr_info)
  print("Average Precisions of Classes:", ap0, ap1, ap2)
  return (ap0+ap1+ap2)/3.0


###########################################
# Test function
###########################################
def test(solo_head, resnet50_fpn, train_loader, test_loader, resume_checkpoint):
  # Create folder for checkpoints
  if IN_COLAB:
    path = os.path.join(HOMEWORK_FOLDER, 'checkpoints')
  else:
    path = os.path.join('.', 'checkpoints')
  os.makedirs(path, exist_ok=True)
  
  # Get the device
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize network
  solo_head=solo_head.to(device)

  # Load the checkpoint
  checkpoint = torch.load(
      os.path.join(path, resume_checkpoint),
      map_location=device
  )
  solo_head.load_state_dict(checkpoint['model_state_dict'])
  

  # Ready the network for inference
  solo_head.eval()


  # P/R variables
  mean_avg_precisions = []
  total_positives = [torch.zeros((1,)).to(device), torch.zeros((1,)).to(device), torch.zeros((1,)).to(device)]
  running_pr_info = [[], [], []]


  for batch_idx, data in enumerate(test_loader):
    # Get raw data
    img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

    # Get FPN feat from resnet
    backout = resnet50_fpn(img)
    fpn_feat_list = list(backout.values())

    # Get the predictions by doing a forward pass
    cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=True)

    # Get the ground truth from target
    ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                   bbox_list,
                                                                   label_list,
                                                                   mask_list)

    # Calculate PR info
    calculate_pr_info(total_positives, running_pr_info, ins_pred_list, ins_gts_list, cate_pred_list, cate_gts_list)

    # Print batch status
    print('Test: [{}/{} ({:.0f}%)]'.format(
    batch_idx * len(img), len(test_loader.dataset),
    100. * batch_idx / len(test_loader)))

    # Post process the predictions
    ori_size = (800, 1088)
    NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(ins_pred_list, cate_pred_list, ori_size)

    # Plot the inference images as we go
    mask_color_list = ["jet", "ocean", "Spectral"]
    solo_head.PlotInfer(
      NMS_sorted_scores_list,
      NMS_sorted_cate_label_list,
      NMS_sorted_ins_list,
      mask_color_list,
      img,
      batch_idx
    )

  precisions0, recalls_0, auc_0 = average_precision(0, total_positives, running_pr_info, return_curves = True)
  precisions1, recalls_1, auc_1 = average_precision(1, total_positives, running_pr_info, return_curves = True)
  precisions2, recalls_2, auc_2 = average_precision(2, total_positives, running_pr_info, return_curves = True)
  mAP = mean_average_precision(total_positives, running_pr_info)

  # Print final status
  print(
      "Finished running inference on all test images"
  )

  # Show P/R information & plots

  # Setup the plot
  plt.figure(figsize=(7,15))
  print("Mean Average Precision over Test Inference:", mAP)
  ax = plt.subplot(3, 1, 1)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 1)", "Recall", "Precision", precisions0, recalls_0)
  ax = plt.subplot(3, 1, 2)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 2)", "Recall", "Precision", precisions1, recalls_1)
  ax = plt.subplot(3, 1, 3)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  plot_array(ax, "P/R Curve (Class = 3)", "Recall", "Precision", precisions2, recalls_2)

  plt.show()


###########################################
# Main
###########################################
if __name__ == '__main__':
    print("Building the dataset...")
    solo_head, resnet50_fpn, train_loader, test_loader = build_dataset()
    
    # In the last argument of test() put the name of the 
    # checkpoint file you want to load the model from
    print("Running inference on the model...")
    test(solo_head, resnet50_fpn, train_loader, test_loader, 'solo8_epoch_49')