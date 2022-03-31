import os
import preprocess
import config

import matplotlib.pyplot as plt

import torch

from sklearn.model_selection import train_test_split

from torch.autograd import Variable
from torch.nn import Sequential, Conv2d, BatchNorm2d, MSELoss
from torch.optim import Adam

import numpy as np

import torchvision

from train_2 import NeuralNet

PATH = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/HGG"
TEMP = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp"
DICE_FILE = "HGG_dice_cnn.csv"

#import the model
MODEL_PATH = '/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp/1_ckpt_cnn_segmentation.pth'
# RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# def load_model_weights(model, weights):
#
#     model_dict = model.state_dict()
#     weights = {k: v for k, v in weights.items() if k in model_dict}
#     if weights == {}:
#         print('No weight could be loaded..')
#     model_dict.update(weights)
#     model.load_state_dict(model_dict)
#
#     return model
#
#
# # model = torchvision.models.__dict__['neuralnet_pytorch'](pretrained=True)
#
# state = torch.load(MODEL_PATH, map_location='cuda:0')
#
# state_dict = state['state_dict']
# # for key in list(state_dict.keys()):
# #     state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
#
# model = load_model_weights(MODEL_PATH, state_dict)
model = NeuralNet()
model.load_state_dict(torch.load(MODEL_PATH))

# fc = nn.Sequential(
#     Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#     BatchNorm2d(4),
#     Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
#     BatchNorm2d(1),)
#
# model.fc = fc



# if RETURN_PREACTIVATION:
#     model.fc = torch.nn.Sequential()
# # else:
#     model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.cuda()

images = torch.rand((4, 1, 3, 3), device='cuda')

out = model(images)



"""Start of testing I think - this needs to be moved into test.py
Will beed to import test_x variable into test.py too"""
# Calculate predictions
with torch.no_grad():
    output = model(test_x.cuda())

softmax = torch.exp(output).cuda()
softmax = softmax.cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
predictions = torch.from_numpy(predictions).float().unsqueeze(1)

# Validating a slice visually if required
# plt.title("value")
# plt.imshow(val_y[0,0,:,:])
# plt.show()
# plt.title("prediction")
# plt.imshow(predictions[0,0,:,:])
# plt.show()

# Calculate dice score of predictions and save in file
#dice_score = dice(val_y.flatten(), predictions.flatten())
dice_score = 1 - dice_loss(val_y, predictions)

print(f'DICE score for slice: {dice_score}')

slice_dice.append(dice_score)

with open(os.path.join(TEMP, DICE_FILE), "a") as myfile:
    myfile.write(f"{dice_score},\n")
