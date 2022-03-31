# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:44:24 2022

@author: felix, phoenix, katie

Disclaimer: Code is somewhat loosely inspired by pytorch documentation and official examples https://pytorch.org/
as well as by https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
"""

import os
import preprocess
import config

import matplotlib.pyplot as plt

import torch

from sklearn.model_selection import train_test_split

from torch.autograd import Variable
from torch.nn import Sequential, Conv2d, BatchNorm2d, MSELoss
from torch.optim import Adam

#from sklearn.metrics import accuracy_score
# from scipy.spatial.distance import dice

import numpy as np

PATH = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/HGG"
TEMP = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp"
DICE_FILE = "HGG_dice_cnn.csv"
#Save the model
filename_pth = '/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp/'

def dice_loss(preds, targets):
    loss = 0
    smooth = 1.0
    for i in range(preds.shape[0]):
        pred = preds[i,0,:,:]
        target = targets[i,0,:,:]
        intersection = (pred * target).sum()
        dice_score = ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
        loss += 1 - dice_score
    loss = loss/preds.shape[0]
    return loss




class NeuralNet(torch.nn.Module):

    # CNN Definition
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(1)
        )

    # Forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        return x



if __name__ == '__main__':

    # Clear output file
    open(os.path.join(TEMP, DICE_FILE), 'w').close()

    print("Load Data\n=================================================")


    data = []
    counter = 0

    for directory in os.listdir(PATH):
        dir_path = os.path.join(PATH, directory)
        if os.path.isdir(dir_path):

            counter += 1
            print(f'Loading set: {directory}')
            data.append(preprocess.choose_modality(dir_path, [config.ID_LABEL, config.ID_T1], save_checkpoint=TEMP))

            # for testing purposes
            # if counter > 10:
            #     break

    slice_dice = []

    # slice in z-direction, we do a slice by slice approach here, alternatively we could merge all the slices into a single set
    for z in range(data[0][0].shape[2]):
        print(f'Training slice: {z + 1} / {data[0][0].shape[2]}')
        slice_data = []
        slice_label = []
        for (data_label, data_image) in data:
            slice_data.append(data_image[:,:,z])
            slice_label.append(data_label[:,:,z])



        data_x = np.array(slice_data)
        data_y = np.array(slice_label)

        # test split
        data_x, test_x, data_y, test_y = train_test_split(data_x, data_y, test_size = 0.2)

        # validation train / split
        train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size = 0.1)


        train_x  = torch.from_numpy(train_x).float()
        train_y = torch.from_numpy(train_y).float()
        val_x  = torch.from_numpy(val_x).float()
        val_y = torch.from_numpy(val_y).float()
        test_x  = torch.from_numpy(test_x).float()
        test_y = torch.from_numpy(test_y).float()

        # Formatting
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
        val_x = val_x.unsqueeze(1)
        val_y = val_y.unsqueeze(1)
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)

        # For visualisation
        # plt.imshow(val_y[0,0,:,:])
        # plt.show()

        """Model Initialisation Stuff"""
        # Define model, optimizer, and criterion
        model = NeuralNet()
        optimizer = Adam(model.parameters(), lr=0.07)

        # either MSELoss or dice
        #criterion = MSELoss()
        criterion = dice_loss

        # GPU settings
        if torch.cuda.is_available():
            model = model.cuda()
            #criterion = criterion.cuda()

        """This should really be in a separate class above main and called on"""
        def train(epoch):
            model.train()
            # Formatting
            x_train, y_train = Variable(train_x), Variable(train_y)
            x_val, y_val = Variable(val_x), Variable(val_y)
            # If you want to run on your GPU, it's false for my machine though
            if torch.cuda.is_available():
                x_train = x_train.cuda()
                y_train = y_train.cuda()
                x_val = x_val.cuda()
                y_val = y_val.cuda()

            # Clearing optimizer
            optimizer.zero_grad()

            # Apply model
            output_train = model(x_train)
            output_val = model(x_val)

            # Compute loss
            loss_train = criterion(output_train, y_train)
            loss_val = criterion(output_val, y_val)
            train_losses.append(loss_train)
            val_losses.append(loss_val)

            # Update model
            loss_train.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {loss_val}')

        n_epochs = 30
        train_losses = []
        val_losses = []

        # Actual loop for training
        for epoch in range(n_epochs):
            train(epoch)'

        # print(model.state_dict().keys())
        # save all the models per slice
        torch.save(model.state_dict(), filename_pth + str(z) + "_ckpt_cnn_segmentation.pth")
        print(model.state_dict()) ## FOR DEBUGGING THE STATE_DICT
        model = model.cuda()
