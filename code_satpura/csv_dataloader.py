from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image
import os
import numpy as np
import glob
import random
import openslide
import torch
import pandas as pd


# custom_mnist_from_csv = CustomDatasetFromCSV('../data/mnist_in_csv.csv', 28, 28, transformations)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        # print(len(self.data.index))
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.x = np.asarray(self.data.iloc[:, 5])
        self.y = np.asarray(self.data.iloc[:, 6])
        self.wsi_path = np.asarray(self.data.iloc[:, 4])
        self.patch =np.asarray(self.data.iloc[:, 3])
        self.level =np.asarray(self.data.iloc[:, 2])
        # self.height = height
        # self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        level = np.asarray(self.data.iloc[index][2])
        patch = np.asarray(self.data.iloc[index][3])
        wsi_path = np.asarray(self.data.iloc[index][4])
        x = np.asarray(self.data.iloc[index][5])
        y = np.asarray(self.data.iloc[index][6])

        ### added for inceptionv3#############
        if patch == 256: patch = 299
        else: patch = 299*2
        #######################################
        
        #####for resnet#####
        # if patch == 256: patch = 224
        # else: patch = 224*2

        wsi_path = '/home/Drive2/DATA_LUNG_TCGA/' + str(wsi_path)
        wsi = openslide.OpenSlide(wsi_path)
        pil = wsi.read_region((x, y), level, (patch, patch))
        pil = pil.convert('RGB')
        sample = self.transforms(pil)

	# # Convert image from numpy array to PIL image, mode 'L' is for grayscale
 #        img_as_img = Image.fromarray(img_as_np)
 #        img_as_img = img_as_img.convert('L')
 #        # Transform image to tensor
 #        if self.transforms is not None:
 #            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (sample, label)

    def __len__(self):
        # print(len(self.data.index))
        return len(self.data.index)

# train_transforms = transforms.Compose([
#         transforms.Resize(299),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
# custom_from_csv = CustomDatasetFromCSV('data_adeno.csv', train_transforms)
