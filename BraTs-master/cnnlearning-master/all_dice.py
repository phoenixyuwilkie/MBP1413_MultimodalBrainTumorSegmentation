# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:33:51 2022

@author: felix, phoenix, katie

This generates figures used in data analysis
"""

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def dice(a, b):
    smooth = 1.0
    intersection = (a * b).sum()
    dice_score = ((2.0 * intersection + smooth) / (a.sum() + b.sum() + smooth))
    return dice_score

def run(model, grade):
    MODEL = model
    GRADE = grade
    # model options = unet, pspnet_res50, pspnet_res34, pspnet_res18
    # grade options = HGG or LGG
    PATH_PRED = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/pytorch/output/test_"+GRADE+"_prediction_"+MODEL
    PATH_LABEL = "/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/test/"+GRADE+"/label"

    PATH_CSV = GRADE+"_DICE_"+MODEL+".csv"
    PATH_FIGURE = GRADE+"_AUC_"+MODEL+".jpg"

    open(PATH_CSV, 'w').close()

    scores = []

    cnt2 = 0

    threshold_range = np.linspace(0,1,100)
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    for val in threshold_range:
        TP[val] = 0
        FP[val] = 0
        TN[val] = 0
        FN[val] = 0

    # calculations per patient
    for patient in os.listdir(PATH_PRED):
        print(f'{cnt2+1}/{len(os.listdir(PATH_PRED))}')
        cnt = 0
        path_pred_patient = os.path.join(PATH_PRED, patient)
        path_label_patient = os.path.join(PATH_LABEL, patient)
        for file in os.listdir(path_pred_patient):
            if file in os.listdir(path_label_patient):
                cnt += 1
                image_pred = Image.open(os.path.join(path_pred_patient, file))
                r, g, b  = image_pred.split()
                v = (1.5 * (np.array(r) - np.array(g))) / 255 # prediction
                image_label = Image.open(os.path.join(path_label_patient, file))
                grey_scale  = np.array(image_label.split()[0]) / 255 # ground truth
                score = dice(v, grey_scale)
                scores.append(score)

                for val in threshold_range:
                    if np.sum(grey_scale) == 0.0:
                        if score < val:
                            FP[val] += 1
                        else:
                            TN[val] += 1
                    else:
                        if score < val:
                            FN[val] += 1
                        else:
                            TP[val] += 1

                with open(PATH_CSV, "a") as myfile:
                    myfile.write(f"{patient}, {cnt}, {score},\n")
        cnt2 += 1

    print(f'Dice average: {np.average(scores)}')

    TPR = []
    FPR = []

    # print everything that outputs out into csv files
    stdoutOrigin=sys.stdout
    sys.stdout = open(str(GRADE)+"_"+str(MODEL)+"_coords.csv", "w")

    print("val,TP,FP,TN,FN")
    for val in threshold_range:
        if TP[val] + FN[val] == 0 or TN[val] + FP[val] == 0:
            continue
        TPR.append(TP[val] / (TP[val] + FN[val]))
        # FPR.append(1 - FP[val] / (TN[val] + FP[val]))
        FPR.append(FP[val] / (TN[val] + FP[val]))

        print(f'{val}, {TP[val]}, {FP[val]}, {TN[val]}, {FN[val]}') # save this to csv

    # print everything that outputs out into csv files
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    stdoutOrigin2=sys.stdout
    sys.stdout = open(str(GRADE)+"_"+str(MODEL)+"_fig.csv", "w")

    print("threshold_range,TPR")
    for i in range(len(threshold_range)):
        print(str(threshold_range[i])+','+str(TPR[i]))

    sys.stdout.close()
    sys.stdout=stdoutOrigin2

    # pseudo ROC curve plots

    # plt.scatter(FPR, TPR, color='r')
    # plt.plot([0, 1], [0, 1], color='b')
    # plt.title("Pseudo-ROC Curve")
    # plt.xlim(0, 1.1)
    # plt.ylim(0, 1.1)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.savefig(PATH_FIGURE)
    # plt.show()

    y_avg = [np.mean(TPR)] * len(TPR)
    stdv = [np.std(TPR)] * len(TPR)
    # print(y_avg)
    # print(stdv)

    fig = plt.scatter(threshold_range, TPR, label='TPR')
    # add mean and stdv lines
    fig = plt.plot(y_avg, color='green', label='mean')
    fig = plt.plot(stdv, color='red', label='stdv')
    fig = plt.xlabel("Threshold Range")
    fig = plt.ylabel("True Positive Rate")
    fig = plt.xlim(0, 1)
    fig = plt.legend()
    fig = plt.title("True Positive Rate per Threshold value for " + str(GRADE) + " " + str(MODEL) + " model")
    fig = plt.savefig(PATH_FIGURE)
    fig = plt.show(block=False)
    fig = plt.pause(1)
    fig = plt.close()
