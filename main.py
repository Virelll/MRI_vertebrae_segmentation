import torch

import torchvision
import torchvision.transforms as transforms
import pandas as pd
from skimage.util import montage
import copy

import torch.nn.functional as F


# from tensorflow import keras


from torchsummary import summary

import math
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Unet import UNet
from load_data import load_data
from split_data import train_test_split_data
from parameters import *
from dataloader import *
from Net import *
print("Load imports successful\n")

images_384, masks_384, images_320, masks_320 = load_data()
print("Load data successful\n")

img_train_384, img_test_384, masks_train_384, masks_test_384 = train_test_split_data(images_384, masks_384, 384)
print("Split data 384 successful\n")
img_train_320, img_test_320, masks_train_320, masks_test_320 = train_test_split_data(images_320, masks_320, 320)
print("Split data 320 successful\n")

num_classes = count_classes_mask(masks_384)
print("Count classes mask successful\n")

# train_dataloader = createDataLoader(img_train_384,masks_train_384,train_transform,True,num_classes)
# print("Create train dataloader successful\n")
# test_dataloader = createDataLoader(img_test_384,masks_test_384,test_transform,False,num_classes)
# print("Create test dataloader successful\n")

train_dataloader = createCatDataLoader([img_train_384,img_train_320],[masks_train_384,masks_train_320],train_transform,True,num_classes)
print("Create train cat dataloader successful\n")
test_dataloader = createCatDataLoader([img_test_384,img_test_320],[masks_test_384,masks_test_320],test_transform,False,num_classes)
print("Create test cat dataloader successful\n")
print(len(train_dataloader))
print(len(test_dataloader))

optimizer='Adamax'
lr=0.001
epochs=150

model = Net(UNet(num_classes),optimizer=optimizer,lr=lr,epochs=epochs)
print("Create Net successful\n")
model.train(train_dataloader,test_dataloader)
print("Training model successful\n")

model.save_model("D:\\MRI\\All_data_model_FullNet_"+str(optimizer)+"_"+str(lr)+"_Dice_Seg_"+str(epochs)+"_scheduler(maxDice)_patience5_factor0.5.pt")
# model = torch.load("D:\\MRI\\C_model.pt").to(device)

fig1, ax1 = plt.subplots(figsize=(7,7))
fig2, ax2 = plt.subplots(figsize=(7,7))
fig3, ax3 = plt.subplots(figsize=(7,7))


for i in range(len(model.history['train_dice_list'])):
    model.history['train_loss_list'][i] = model.history['train_loss_list'][i].cpu().detach().numpy()
    model.history['train_loss2_list'][i] = model.history['train_loss2_list'][i].cpu().detach().numpy()

    model.history['test_loss_list'][i] = model.history['test_loss_list'][i].cpu().detach().numpy()
    model.history['test_loss2_list'][i] = model.history['test_loss2_list'][i].cpu().detach().numpy()

ax1.plot(model.history['train_dice_list'], label='train_dice_list')
ax1.plot(model.history['test_dice_list'], label='test_dice_list')

ax2.plot(model.history['train_loss_list'], label='train_loss_list')
ax2.plot(model.history['train_loss1_list'], label='train_loss1_list')
ax2.plot(model.history['train_loss2_list'], label='train_loss2_list')

ax3.plot(model.history['test_loss_list'], label='test_loss_list')
ax3.plot(model.history['test_loss1_list'], label='test_loss1_list')
ax3.plot(model.history['test_loss2_list'], label='test_loss2_list')

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()