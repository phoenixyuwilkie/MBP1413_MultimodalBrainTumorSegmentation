# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:28:52 2022

@author: felix, phoenix, katie

This generates most of the figures we used for data analysis
"""

import os
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from all_dice import run

# figure names here + locations as needed
PATH_FIGURE_1 = "TPR"
PATH_FIGURE_2 = "dicePerSlice1"
PATH_FIGURE_3 = "dicePerSlice2"
PATH_FIGURE_4 = "dicePerPatient1"
PATH_FIGURE_5 = "dicePerPatient2"


# comment this out if only needing the last figures
# This grabs individual figures from all_dice
run("unet", "HGG")
run("unet", "LGG")
run("pspnet_res50", "HGG")
run("pspnet_res50", "LGG")
run("pspnet_res34", "HGG")
run("pspnet_res34", "LGG")
run("pspnet_res18", "HGG")
run("pspnet_res18", "LGG")

# csv Initialisation
lgg_unet = "LGG_unet_fig.csv"
lgg_res50 = "LGG_pspnet_res50_fig.csv"
hgg_unet = "HGG_unet_fig.csv"
hgg_res50 = "HGG_pspnet_res50_fig.csv"
hgg_res34 = "HGG_pspnet_res34_fig.csv"
lgg_res34 = "HGG_pspnet_res34_fig.csv"
hgg_res18 = "HGG_pspnet_res18_fig.csv"
lgg_res18 = "HGG_pspnet_res18_fig.csv"

lgg_cnn = "LGG_dice_cnn.csv"
hgg_cnn = "HGG_dice_cnn.csv"


df1 = pd.read_csv(hgg_res50)
df2 = pd.read_csv(lgg_res50)
df3 = pd.read_csv(hgg_unet)
df4 = pd.read_csv(lgg_unet)
df5 = pd.read_csv(hgg_res18)
df6 = pd.read_csv(lgg_res18)
df7 = pd.read_csv(hgg_res34)
df8 = pd.read_csv(lgg_res34)

cnnhgg = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp/HGG_dice_cnn.csv")
cnnlgg = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/tmp/LGG_dice_cnn.csv")

lgg1 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/LGG_DICE_unet.csv")
lgg2 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/LGG_DICE_pspnet_res50.csv")
lgg3 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/LGG_DICE_pspnet_res34.csv")
lgg4 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/LGG_DICE_pspnet_res18.csv")

hgg1 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/HGG_DICE_unet.csv")
hgg2 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/HGG_DICE_pspnet_res50.csv")
hgg3 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/HGG_DICE_pspnet_res34.csv")
hgg4 = pd.read_csv("/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/BraTs-master/cnnlearning-master/HGG_DICE_pspnet_res18.csv")

# this one for Unet only to show how bad it is
fig4 = plt.plot(lgg1.iloc[:154,1], lgg1.iloc[:154,2])
fig4 = plt.ylabel("DICE using LGG trained Unet")
fig4 = plt.xlabel("Slice")
fig4 = plt.title("DICE per slice of patient Brats18_2013_0_1 for LGG")
fig4 = plt.savefig('LGG_Unet_DICE.png')
fig4 = plt.show()
fig4 = plt.pause(1)
fig4 = plt.close()

fig5 = plt.plot(hgg1.iloc[:154,1], hgg1.iloc[:154,2])
fig5 = plt.ylabel("DICE using HGG trained Unet")
fig5 = plt.xlabel("Slice")
fig5 = plt.title("DICE per slice of patient Brats18_CBICA_ASV_1 for HGG")
fig5 = plt.savefig('HGG_Unet_DICE.png')
fig5 = plt.show()
fig5 = plt.pause(1)
fig5 = plt.close()

# this one for all resnets - get average per patient
lgg1_avg = lgg1.groupby([lgg1.iloc[:,0]]).mean()
avgl1 = lgg1_avg.iloc[:,1]
lgg2_avg = lgg2.groupby([lgg2.iloc[:,0]]).mean()
avgl2 = lgg2_avg.iloc[:,1]
lgg3_avg = lgg3.groupby([lgg3.iloc[:,0]]).mean()
avgl3 = lgg3_avg.iloc[:,1]
lgg4_avg = lgg4.groupby([lgg4.iloc[:,0]]).mean()
avgl4 = lgg4_avg.iloc[:,1]

hgg1_avg = hgg1.groupby([hgg1.iloc[:,0]]).mean()
avgh1 = hgg1_avg.iloc[:,1]
hgg2_avg = hgg2.groupby([hgg2.iloc[:,0]]).mean()
avgh2 = hgg2_avg.iloc[:,1]
hgg3_avg = hgg3.groupby([hgg3.iloc[:,0]]).mean()
avgh3 = hgg3_avg.iloc[:,1]
hgg4_avg = hgg4.groupby([hgg4.iloc[:,0]]).mean()
avgh4 = hgg4_avg.iloc[:,1]

# histograms -- unfortunately doesn't show how bad the unet is
# print(avgl1["Brats18_2013_0_1"])
labels = ["Brats18_2013_0_1","Brats18_TCIA09_141_1","Brats18_TCIA09_402_1","Brats18_TCIA10_103_1","Brats18_TCIA10_130_1","Brats18_TCIA10_307_1","Brats18_TCIA10_346_1","Brats18_TCIA10_408_1","Brats18_TCIA10_490_1","Brats18_TCIA10_640_1","Brats18_TCIA13_618_1","Brats18_TCIA13_630_1","Brats18_TCIA13_634_1","Brats18_TCIA13_642_1","Brats18_TCIA13_653_1"]

width = 0.2
hist = plt.bar(np.arange(len(avgl1)), avgl1, width=width)
hist = plt.bar(np.arange(len(avgl2))+ width, avgl2, width=width)
hist = plt.bar(np.arange(len(avgl3))+ 2*width, avgl3, width=width)
hist = plt.bar(np.arange(len(avgl4))+ 3*width, avgl4, width=width)
hist = plt.legend(["unet","pspnet res50","pspnet res34","pspnet res18"])
hist = plt.xlabel("Patient")
hist = plt.ylabel("Mean DICE score per patient")
hist = plt.xticks(np.arange(len(labels)),labels, rotation='vertical')
hist = plt.title("DICE per patient with different model comparisons for LGG")
hist = plt.savefig(PATH_FIGURE_4)
hist = plt.show()
hist = plt.pause(1)
hist = plt.close()

labels2 = ["Brats18_2013_12_1","Brats18_2013_21_1","Brats18_2013_27_1","Brats18_2013_2_1","Brats18_2013_5_1","Brats18_CBICA_ABE_1","Brats18_CBICA_ABO_1","Brats18_CBICA_AME_1","Brats18_CBICA_ANP_1","Brats18_CBICA_AOO_1","Brats18_CBICA_APR_1","Brats18_CBICA_AQG_1","Brats18_CBICA_AQQ_1","Brats18_CBICA_AQZ_1","Brats18_CBICA_ASG_1","Brats18_CBICA_ASU_1","Brats18_CBICA_ASV_1","Brats18_CBICA_AUQ_1","Brats18_CBICA_AWG_1","Brats18_CBICA_AXO_1","Brats18_CBICA_BFB_1","Brats18_TCIA01_180_1","Brats18_TCIA01_203_1","Brats18_TCIA01_390_1","Brats18_TCIA01_412_1","Brats18_TCIA01_499_1","Brats18_TCIA02_118_1","Brats18_TCIA02_198_1","Brats18_TCIA02_300_1","Brats18_TCIA02_370_1","Brats18_TCIA02_430_1","Brats18_TCIA02_491_1","Brats18_TCIA02_608_1","Brats18_TCIA03_265_1","Brats18_TCIA04_111_1","Brats18_TCIA04_437_1","Brats18_TCIA05_444_1","Brats18_TCIA06_165_1","Brats18_TCIA06_211_1","Brats18_TCIA06_603_1","Brats18_TCIA08_167_1","Brats18_TCIA08_278_1"]

width = 0.2
hist2 = plt.bar(np.arange(len(avgh1)), avgh1, width=width)
hist2 = plt.bar(np.arange(len(avgh2))+ width, avgh2, width=width)
hist2 = plt.bar(np.arange(len(avgh3))+ 2*width, avgh3, width=width)
hist2 = plt.bar(np.arange(len(avgh4))+ 3*width, avgh4, width=width)
hist2 = plt.legend(["unet","pspnet res50","pspnet res34","pspnet res18"])
hist2 = plt.xlabel("Patient")
hist2 = plt.ylabel("Mean DICE score per patient")
hist2 = plt.xticks(np.arange(len(labels2)),labels2, rotation='vertical')
hist2 = plt.title("DICE per patient with different model comparisons for HGG")
hist2 = plt.savefig(PATH_FIGURE_5)
hist2 = plt.show()
hist2 = plt.pause(1)
hist2 = plt.close()

# data = [avgh1,avgh2,avgh3,avgh4]
# print(data)

# dice for every slice
fig2 = plt.scatter(cnnhgg,cnnlgg)
fig2 = plt.xlabel("CNN DICE score per slice for HGG")
fig2 = plt.ylabel("CNN DICE score per slice for LGG")
fig2 = plt.title("DICE score per slice of HGG vs LGG")
fig2 = plt.savefig(PATH_FIGURE_2)
fig2 = plt.show()
fig2 = plt.pause(1)
fig2 = plt.close()

# dice for every slice
fig3 = plt.scatter(cnnhgg.index,cnnhgg.iloc[:,0])
fig3 = plt.scatter(cnnlgg.index,cnnlgg.iloc[:,0])
fig3 = plt.xlabel("Slice number")
fig3 = plt.ylabel("CNN DICE score per slice")
fig3 = plt.legend(["CNN for HGG","CNN for LGG"])
fig3 = plt.title("DICE score per slice of HGG and LGG")
fig3 = plt.savefig(PATH_FIGURE_3)
fig3 = plt.show()
fig3 = plt.pause(1)
fig3 = plt.close()

# models we tried
hgg_auc_res50 = sum(df1.iloc[:,1])/len(df1.iloc[:,1])
print("HGG PSPNet ResNet50 AUC: "+str(hgg_auc_res50))
lgg_auc_res50 = sum(df2.iloc[:,1])/len(df2.iloc[:,1])
print("LGG PSPNet ResNet50 AUC: "+str(lgg_auc_res50))
hgg_auc_unet = sum(df3.iloc[:,1])/len(df3.iloc[:,1])
print("HGG UNet AUC: "+str(hgg_auc_unet))
lgg_auc_unet = sum(df4.iloc[:,1])/len(df4.iloc[:,1])
print("LGG UNet AUC: "+str(lgg_auc_unet))
hgg_auc_res18 = sum(df5.iloc[:,1])/len(df5.iloc[:,1])
print("HGG PSPNet ResNet18 AUC: "+str(hgg_auc_res18))
lgg_auc_res18 = sum(df6.iloc[:,1])/len(df6.iloc[:,1])
print("LGG PSPNet ResNet18 AUC: "+str(lgg_auc_res18))
hgg_auc_res34 = sum(df7.iloc[:,1])/len(df7.iloc[:,1])
print("HGG PSPNet ResNet34 AUC: "+str(hgg_auc_res34))
lgg_auc_res34 = sum(df8.iloc[:,1])/len(df8.iloc[:,1])
print("LGG PSPNet ResNet34 AUC: "+str(lgg_auc_res34))

# this graph is in paper
fig = plt.scatter(df1.iloc[:,0], df1.iloc[:,1], s=3)
fig = plt.scatter(df2.iloc[:,0], df2.iloc[:,1], s=3)
fig = plt.scatter(df3.iloc[:,0], df3.iloc[:,1], s=3)
fig = plt.scatter(df4.iloc[:,0], df4.iloc[:,1], s=3)
fig = plt.scatter(df5.iloc[:,0], df5.iloc[:,1], s=3)
fig = plt.scatter(df6.iloc[:,0], df6.iloc[:,1], s=3)
fig = plt.scatter(df7.iloc[:,0], df7.iloc[:,1], s=3)
fig = plt.scatter(df8.iloc[:,0], df8.iloc[:,1], s=3)
fig = plt.xlabel("Threshold Range")
fig = plt.ylabel("True Positive Rate")
fig = plt.legend(["HGG PSPNet ResNet50","LGG PSPNet ResNet50","HGG UNet", "LGG UNet", "HGG PSPNet ResNet18", "LGG PSPNet ResNet18", "HGG PSPNet ResNet34", "LGG PSPNet ResNet34"])
fig = plt.title("True Positive Rate per Threshold value for every compared model")
fig = plt.savefig(PATH_FIGURE_1)
fig = plt.show()
fig = plt.pause(1)
fig = plt.close()
