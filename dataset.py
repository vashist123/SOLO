## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0

#################
# Detect CoLab
#################
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
      images_h5 = h5py.File(paths[0], mode = 'r')
      masks_h5 = h5py.File(paths[1], mode = 'r')
      self.images = images_h5.get(f'{list(images_h5.keys())[0]}')[()]
      self.masks = masks_h5.get(f'{list(masks_h5.keys())[0]}')[()]
      self.labels = np.load(paths[2],allow_pickle=True)
      self.bboxes = np.load(paths[3],allow_pickle=True)
      self.new_masks = []
      j=0
      for i in range(len(self.labels)):
        self.new_masks.append(self.masks[j:j+len(self.labels[i])])
        j = j + len(self.labels[i])
      # self.new_masks = self.new_masks

    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, idx):

      img = self.images[idx]
      mask = self.new_masks[idx]
      bbox = self.bboxes[idx]
      label = self.labels[idx]
      transed_img,transed_mask,transed_bbox = self.pre_process_batch(img,mask,bbox)


        # check flag
      assert transed_img.shape == (3, 800, 1088)
      assert transed_bbox.shape[0] == transed_mask.shape[0]
      return transed_img, label, transed_mask, transed_bbox
    def __len__(self):
      return len(self.images)

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.toTensor = transforms.ToTensor()
      img = img.astype('float32')
      img = self.toTensor(img)  
      img = img.to(device)
      img = img.permute(1,2,0)
      img = img.unsqueeze(dim=0)
      img = F.interpolate(img,(800,1066), mode = 'bilinear', align_corners=False)
      img = img.squeeze(dim=0)
      img = transforms.functional.normalize(img,(0.485,0.456,0.406),(0.229,0.224,0.225))
      img = F.pad(img,(11,11))


      mask = mask.astype('float32')
      mask = self.toTensor(mask)
      mask = mask.to(device)
      mask = mask.permute(1,2,0)
      mask = mask.unsqueeze(dim=0)
      mask = F.interpolate(mask,(800,1066), mode = 'bilinear', align_corners=False)
      mask = mask.squeeze(dim=0)
      mask = F.pad(mask,(11,11))
      # mask = 1-mask 

      bbox1 = np.zeros(bbox.shape)
      for i in range(len(bbox)):
        bbox1[i][0] = bbox[i][0]*800/300 + 11
        bbox1[i][1] = bbox[i][1]*1066/400 
        bbox1[i][2] = bbox[i][2]*800/300 + 11
        bbox1[i][3] = bbox[i][3]*1066/400


      # check flag
      assert img.shape == (3, 800, 1088)
      assert bbox.shape[0] == mask.shape[0]
      return (img, mask, bbox)



class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []

        for transed_img,label,transed_mask,transed_bbox in batch:
          transed_img_list.append(transed_img) 
          label_list.append(label)
          transed_mask_list.append(transed_mask)
          transed_bbox_list.append(transed_bbox)

        return torch.stack(transed_img_list,dim=0),label_list,transed_mask_list,transed_bbox_list

    def loader(self):
      return DataLoader(dataset=self.dataset,batch_size = self.batch_size,shuffle = self.shuffle,num_workers = self.num_workers,collate_fn = self.collect_fn)

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
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

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]

        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        # label = [label_img.to(device) for label_img in label]
        # mask = [mask_img.to(device) for mask_img in mask]
        # bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
          labels_for_image = label[i] 
          plt.figure()
          plot_img = img[i]
          plot_img = transforms.functional.normalize(plot_img,(-0.485/0.229,-0.456/0.224,-0.406/0.225),(1/0.229,1/0.224,1/0.225))
          plot_img = plot_img.permute(1,2,0).type(torch.long)
          plt.imshow(plot_img)

          mask_num = mask[i].shape[0]

          if mask_num>1:
            for j in range(mask_num):
              label_for_bbox = labels_for_image[j]
              mask_init = mask[i][j]
              mask_init = np.reshape(mask_init,(800,1088,1))
              mask_init = np.squeeze(mask_init)
              mask_new = np.ma.masked_where(mask_init==0,mask_init)
              plt.imshow(mask_new,cmap=mask_color_list[label_for_bbox-1], alpha=0.4)

          else:
            label_for_bbox = labels_for_image[0]
            mask_init = mask[i]
            mask_init = np.reshape(mask_init,(800,1088,1))
            mask_init=np.squeeze(mask_init)
            mask_new = np.ma.masked_where(mask_init==0,mask_init)
            plt.imshow(mask_new,cmap=mask_color_list[label_for_bbox-1], alpha=0.4)

          ax = plt.gca()
          for j in range(len(label[i])):
            rect2 = patches.Rectangle((bbox[i][j][0],bbox[i][j][1]),bbox[i][j][2] - bbox[i][j][0],bbox[i][j][3] - bbox[i][j][1],linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect2)
          plt.show()

        if iter == 10:
            break

