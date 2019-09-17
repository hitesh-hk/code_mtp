from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data_loader import wsitxt_train
from tensorboardX import SummaryWriter
import pandas as pd

val_interval = 10
runs_path='/home/hitesh/MTP/'
writer = SummaryWriter()
# writer = SummaryWriter(runs_path)

train_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
val_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '/home/Drive2/DATA_LUNG_TCGA/data'

train_dataset = wsitxt_train(os.path.join(data_dir, 'train'), train_transforms)
val_dataset = wsitxt_train(os.path.join(data_dir, 'val'), val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=25):
    iter_count = 0
    iter_inc=0
    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        running_corrects_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # prediction = pd.DataFrame(outputs, columns=['outputs','probabilities']).to_csv('prediction.csv')
            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)

            if iter_count % val_interval == 0:
                model.eval()
                
                with torch.no_grad():
                    running_loss_val = 0.0
                    running_corrects_val = 0

                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        running_loss_val += loss.item() * inputs.size(0)
                        running_corrects_val += torch.sum(preds == labels.data)

                    epoch_loss_val = running_loss_val / len(val_dataset)
                    epoch_acc_val = running_corrects_val.double() / len(val_dataset)
                    
                    writer.add_scalar('VAL_LOSS', epoch_loss_val, iter_count/val_interval)
                    writer.add_scalar('VAL_ACCU', epoch_acc_val, iter_count/val_interval)
                model.train()
                torch.save(model.state_dict(), '/home/Drive2/hitesh/saved_models/'+str(iter_inc)+'.pth')
                iter_inc=iter_inc+1

            iter_count += 1
            
        epoch_loss_train = running_loss_train / len(train_dataset)
        epoch_acc_train = running_corrects_train.double() / len(train_dataset)

        writer.add_scalar('TRN_LOSS', epoch_loss_train, epoch)
        writer.add_scalar('TRN_ACCU', epoch_acc_train, epoch)

        # torch.save(model.state_dict(), '/home/hitesh/MTP/saved_models/frozen/adsq/'+str(epoch)+'.pth')



####for inception
model_ft = models.inception_v3(pretrained=True)
model_ft.aux_logits=False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)


###for resnet18
# model_ft = models.resnet34(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)

#for loading saved model
# model = torch.load('/home/hitesh/MTP/saved_models_inceptionv3/frozen/adsq/250.pth')
# # model.load_state_dict(torch.load(PATH))
# model.eval()


model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, weight_decay=1e-3)

model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=100)
