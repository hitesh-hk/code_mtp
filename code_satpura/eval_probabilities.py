from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data_loader_eval import wsitxt_test
from tensorboardX import SummaryWriter
import pandas as pd

data_dir = '/home/Drive2/DATA_LUNG_TCGA/data'
description='more details on experiment can be provided here'
writer = SummaryWriter('runs/exp-1',comment='Heat Maps',filename_suffix='_trial')
val_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_dataset = wsitxt_test(os.path.join(data_dir, 'val'), val_transforms)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
print(len(val_loader))
# exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


#for loading saved model
model = models.inception_v3(pretrained=True)
model.aux_logits=False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# print(model.keys())
model.load_state_dict(torch.load('/home/Drive2/hitesh/saved_models_inceptionv3/frozen/adsq/249.pth'))

model.to(device)
model.eval()

val_interval=10
iter_count = 0
running_loss_val = 0.0
running_corrects_val = 0
with torch.no_grad():
    for inputs, labels, x, y, patch ,wsi_dimensions_0,wsi_dimensions_1 in val_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probabs = F.softmax(outputs,dim=1)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # print(probabs.cpu().numpy())
        # print(probabs.cpu().numpy()[1])
        # print(inputs.size())
        # print(probabs.cpu().numpy()[iter_count][0])
        # print(probabs.cpu().numpy()[iter_count][1])
        # print(x, y, patch,preds,probabs.cpu().numpy()[iter_count][0],probabs.cpu().numpy()[iter_count][1] ,wsi_dimensions_0,wsi_dimensions_1)
        # print('x,y:',x.cpu().numpy(),y.cpu().numpy(),patch.cpu().numpy(),preds.cpu().numpy())
        # print(wsi_dimensions_0.cpu().numpy(),wsi_dimensions_1.cpu().numpy())
        # print('------')
        # continue
        # exit()
        # print(probabs.shape)
        prediction = pd.DataFrame({'x':x.cpu().numpy(), 'y':y.cpu().numpy(),'patch':patch.cpu().numpy(),'wsi_dimensions_0':wsi_dimensions_0.cpu().numpy(),'wsi_dimensions_1':wsi_dimensions_1.cpu().numpy(),'predictions':preds.cpu().numpy(),'probability_0':probabs.cpu().numpy()[iter_count][0],'probability_1':probabs.cpu().numpy()[iter_count][1]}).to_csv('prediction.csv',mode='a', header=False)

        # if(iter_count==3):
        #     exit()
        # prediction = pd.DataFrame({'x':x.cpu().numpy(), 'y':y.cpu().numpy(), 'patch':patch.cpu().numpy(),'predictions':preds.cpu().numpy(),'probability_0':probabs.cpu().numpy()[iter_count][0],'probability_1':probabs.cpu().numpy()[iter_count][1] ,'wsi_dimensions_0':wsi_dimensions_0.cpu().numpy(),'wsi_dimensions_1':wsi_dimensions_1.cpu().numpy()}).to_csv('prediction.csv')

        #######for loss calculation and logging########
        # running_loss_val += loss.item() * inputs.size(0)
        # running_corrects_val += torch.sum(preds == labels.data)

        # epoch_loss_val = running_loss_val / len(val_dataset)
        # epoch_acc_val = running_corrects_val.double() / len(val_dataset)

        # writer.add_scalar('VAL_LOSS', epoch_loss_val, iter_count/val_interval)
        # writer.add_scalar('VAL_ACCU', epoch_acc_val, iter_count/val_interval)
        model.eval()

        # iter_count += 1