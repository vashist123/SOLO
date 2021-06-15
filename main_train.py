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
import os
import time



#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

###########################################
# Build dataset
###########################################
def build_dataset():
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
  test_build_loader = BuildDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
  test_loader = test_build_loader.loader()


  resnet50_fpn = Resnet50Backbone()
  solo_head = SOLOHead(num_classes=4) ## class number is 4, because consider the background as one category.
  return solo_head, resnet50_fpn, train_loader, test_loader

###########################################
# Train function
###########################################
def train(solo_head, resnet50_fpn, train_loader, test_loader, resume_checkpoint = None):
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

  # Hyperparameters
  learning_rate = 0.01/8
  weight_decay = .0001
  momentum = .9
  num_epochs = 36

  ## Intialize Optimizer
  optimizer=torch.optim.SGD(solo_head.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

  ## Keep Track of Losses
  losses = []
  focal_losses = []
  dice_losses = []

  if resume_checkpoint:
    checkpoint = torch.load(
        os.path.join(path, resume_checkpoint),
        map_location=device
    )
    solo_head.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']+1
    losses = checkpoint['losses']
    focal_losses = checkpoint['focal_losses']
    dice_losses = checkpoint['dice_losses']
  else:
    epoch = 0
  
  for epoch in range(epoch, num_epochs):
    # Ready the network for training
    solo_head.train()
    
    # Intialize list to hold running losses during batch training
    running_losses = []
    running_focal_losses = []
    running_dice_losses = []


    for batch_idx, data in enumerate(train_loader):
        # Get raw data
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]

        # Get FPN feat from resnet
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())

        # Get the predictions by doing a forward pass
        optimizer.zero_grad()
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)

        # Get the ground truth from target
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)


        # Calculate the loss
        focal_loss, dice_loss, loss = solo_head.loss(
            cate_pred_list,
            ins_pred_list,
            ins_gts_list,
            ins_ind_gts_list,
            cate_gts_list
        )

        # Backprop
        loss.backward()
        optimizer.step()

        # Save the losses for later
        running_losses.append(loss.item())
        running_focal_losses.append(focal_loss.item())
        running_dice_losses.append(dice_loss.item())

        # Print batch status
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(img), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

    # After all batches add losses to the curves 
    losses.extend(running_losses)
    focal_losses.extend(running_focal_losses)
    dice_losses.extend(running_dice_losses)

    # Print epoch status
    print(
        "Epoch:", epoch,
        "Focal Loss:", sum(running_focal_losses) / float(len(running_focal_losses)),
        "Dice Loss:", sum(running_dice_losses) / float(len(running_dice_losses)),
        "Total Loss:", sum(running_losses) / float(len(running_losses)),
    )

    # Save a checkpoint at the end of each epoch
    chkpt_path = os.path.join(path,'solo_epoch_'+str(epoch))
    torch.save({
      'epoch': epoch,
      'model_state_dict': solo_head.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'losses': losses,
      'focal_losses': focal_losses,
      'dice_losses': dice_losses,
    }, chkpt_path)

  return losses, focal_losses, dice_losses

###########################################
# Plotting Loss Curves function
###########################################

def plot_array(ax, title, xlabel, ylabel, y_data, x_data = None):
    if x_data is None:
        ax.plot(y_data)
    else:
        ax.plot(x_data, y_data)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_loss_curves(losses, focal_losses, dice_losses):
    # Setup the plot
    plt.figure(figsize=(7,15))

    # Plot the loss curves
    ax = plt.subplot(3, 1, 1)
    plot_array(ax, "Training Total Loss", "Iteration", "Total Loss", losses)
    ax = plt.subplot(3, 1, 2)
    plot_array(ax, "Training Focal Loss", "Iteration", "Focal Loss", focal_losses)
    ax = plt.subplot(3, 1, 3)
    plot_array(ax, "Training Dice Loss", "Iteration", "Dice Loss", dice_losses)

    # Show the plot
    plt.show()

###########################################
# Main
###########################################
if __name__ == '__main__':
    print("Building the dataset...")
    solo_head, resnet50_fpn, train_loader, test_loader = build_dataset()
    
    print("Training...")
    losses, focal_losses, dice_losses = train(solo_head, resnet50_fpn, train_loader, test_loader)
    
    # Plot the loss curves
    print("Plotting loss curves...")
    plot_loss_curves(losses, focal_losses, dice_losses)

