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

class wsitxt_train(Dataset):
	def __init__(self, path, transforms=None):
		self.classes = os.listdir(path)
		self.path = path
		self.samples = []
		for i, class_ in enumerate(self.classes):
			temp_samples = glob.glob(os.path.join(self.path, class_)+'/*')
			temp_samples = [(s, i) for s in temp_samples]
			self.samples += temp_samples
			self.transforms = transforms
        
	def __getitem__(self, index):
		txt, label = self.samples[index]
		txt = list(open(txt))
		wsi_path= txt[0][:-1]
		level, patch = map(int, txt[1][:-1].split())

		### added for inceptionv3#############
		if patch == 256: patch = 299
		else: patch = 299*2
		#######################################
		
		#####for resnet#####
		# if patch == 256: patch = 224
		# else: patch = 224*2
		sample_lst=[]
		label_lst=[]
		wsi_path = '/home/Drive2/DATA_LUNG_TCGA/' + wsi_path
		wsi = openslide.OpenSlide(wsi_path)
		for h in range(2,len(txt)):

			# print(h)
			# h=2
			x, y = map(int, txt[h][:-1].split('__'))
			pil = wsi.read_region((x, y), level, (patch, patch))
			pil = pil.convert('RGB')
			sample = self.transforms(pil)
			label = torch.tensor(int(label))
			# return sample, label
			sample_lst.append(sample)
			label_lst.append(label)
			


		# x, y = map(int, random.choice(txt[2:])[:-1].split('__'))
		# wsi_path = '/home/Drive2/DATA_LUNG_TCGA/' + wsi_path
		# wsi = openslide.OpenSlide(wsi_path)
		# pil = wsi.read_region((x, y), level, (patch, patch))
		# pil = pil.convert('RGB')
		# sample = self.transforms(pil)
		# label = torch.tensor(int(label))
		# print(len(sample_lst))
		# exit()
		return sample_lst, label_lst

	def __len__(self):
        	return len(self.samples)


#train_transforms = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

#temp_dataset = wsitxt_train('/home/Drive2/DATA_LUNG_TCGA/train_temp', transforms = train_transforms)

#a, b = temp_dataset.__getitem__(3)

#print(b)
