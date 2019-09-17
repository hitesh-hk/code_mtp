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

class wsitxt_test(Dataset):
	def __init__(self, path, transforms=None):
		self.classes = os.listdir(path)
		self.path = path
		self.samples = []
		text_samples=[]
		# self.x_cor=[]
		# self.y_cor=[]
		for i, class_ in enumerate(self.classes):
			# print(i,class_)
			temp_samples = glob.glob(os.path.join(self.path, class_)+'/*')
			temp_samples = [(s, i) for s in temp_samples]
			# for j in range(0,len(temp_samples)):
			for j in range(0,1):

				txt, label = temp_samples[i]
				# print(j)
				# continue
				# print(txt,temp_samples[i])
				txt = list(open(txt))
				wsi_path= txt[0][:-1]
				# print(wsi_path)
				level, patch = map(int, txt[1][:-1].split())
				# print("level,patch",level,patch,wsi_path)

				# print(txt[2:][:-1])
				# z1, z2 = map(int, txt[2][:-1].split('__'))
				# print("z1,z2",z1,z2)
				# x, y = map(int, random.choice(txt[2:])[:-1].split('__'))
				x_cor=[]
				y_cor=[]
				for h in range(2,len(txt)):
							# print(h)
							# h=2
							x, y = map(int, txt[h][:-1].split('__'))
							# print(x,y)
							text_samples.append([wsi_path, level,patch,x,y,label])

				# print("I am x and y",text_samples[0])
				# print(x_cor,y_cor)
				# print(len(temp_samples))
				# print(s,i)
				# print(i,class_)

				self.samples += text_samples
				# self.x_cor =x_cor
				# self.y_cor =y_cor
				self.transforms = transforms
				# return temp_samples,x_cor,y_cor
        
	def __getitem__(self, index):
		# print(index)
		wsi_path, level, patch, x, y, label = self.samples[index]
		# print(len(self.samples))
		# print(wsi_path,level, patch, x, y,label)
		# print("I am in __getitem__")
		# print(label)
		# exit()
		# txt = list(open(txt))
		# wsi_path= txt[0][:-1]
		# level, patch, x, y = map(int, txt[1][:-1].split())

		### added for inceptionv3#############
		if patch == 256:
			patch = 299
		else: 
			patch = 299*2
		#######################################
		# #####for resnet#####
		# if patch == 256: patch = 224
		# else: patch = 224*2
		# x, y = map(int, random.choice(txt[2:])[:-1].split('__'))
		wsi_path = '/home/Drive2/DATA_LUNG_TCGA/' + wsi_path
		wsi = openslide.OpenSlide(wsi_path)

		# print("I have fetched slide",wsi_path)
		# exit()	
		# for k in range(len(x_cor)): 
		pil = wsi.read_region((x, y), level, (patch, patch))
		pil = pil.convert('RGB')
		# print(pil.size)
		sample = self.transforms(pil)
		label = torch.tensor(int(label))
		return sample, label, x, y, patch,wsi.dimensions[0],wsi.dimensions[1]

	def __len__(self):
		# print(len(self.samples))
		return len(self.samples)


#train_transforms = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

#temp_dataset = wsitxt_train('/home/Drive2/DATA_LUNG_TCGA/train_temp', transforms = train_transforms)

#a, b = temp_dataset.__getitem__(3)

#print(b)
