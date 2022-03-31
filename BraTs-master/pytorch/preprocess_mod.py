import cv2
import os
import pdb
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelBinarizer

PATH_MASTER = '/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/LGG'
IMG_OUTPUT_ROOT = '/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/train/LGG/image_T1'
LABEL_OUTPUT_ROOT = '/media/phoenixyuwilkie/Dragonfly_sda2/UofT_MBP_PhD/AI_Class/2020/MICCAI_BraTS_2018_Data_Training/train/LGG/label'

# IMG_ROOT = './Task01_BrainTumor/imagesTr'
# IMG_PATH = './Task01_BrainTumor/imagesTr/BRATS_148.nii.gz'
# IMG_OUTPUT_ROOT = './train/image_T1'

# LABEL_ROOT = './Task01_BrainTumor/labelsTr'
# IABEL_PATH = './Task01_BrainTumor/labelsTr/BRATS_148.nii.gz'
# LABEL_OUTPUT_ROOT = './train/label'

L0 = 0      # Background
L1 = 50     # Necrotic and Non-enhancing Tumor
L2 = 100    # Edema
L3 = 150    # Enhancing Tumor

# MRI Image channels Description
# ch0: FLAIR / ch1: T1 / ch2: T1c/ ch3: T2
# cf) In this project, we use FLAIR and T1c MRI dataset
#
# Data Load Example
#img = nib.load(IMG_PATH)
#img = (img.get_fdata())[:,:,:,3]                # img shape = (240,240,155)


# MRI Label Channels Description
# 0: Background         / 1: Necrotic and non-enhancing tumor (paper, 1+3)
# 2: edema (paper, 2)   / 3: Enhancing tumor (paper, 4)
#
# <Input>           <Prediction>
# FLAIR             Complete(1,2,3)
# FLAIR             Core(1,3)
# T1c               Enhancing(3)
#
# Data Load Example
# label = nib.load(LABEL_PATH)
# label = (label.get_fdata()).astype(np.uint16)   # label shape = (240,240,155)


def nii2jpg_img(img_path, output_root, img_name=None):
    if img_name is None:
        img_name = (img_path.split('/')[-1]).split('.')[0]
    output_path = os.path.join(output_root, img_name)
    try:
        os.makedirs(output_root)
    except:
        pass
    try:
        os.makedirs(output_path)
    except:
        pass
    img = nib.load(img_path)
    #img = (img.get_fdata())[:,:,:,1]
    img = (img.get_fdata())[:,:,:]
    img = (img/img.max())*255
    img = img.astype(np.uint8)

    for i in range(img.shape[2]):
        filename = os.path.join(output_path, img_name+'_'+str(i)+'.jpg')
        gray_img = img[:,:,i]
        #color_img = np.expand_dims(gray_img, 3)
        #color_img = np.concatenate([color_img, color_img, color_img], 2)

        # COLOR LABELING
        #c255 = np.expand_dims(np.ones(gray_img.shape)*255, 3)
        #c0 = np.expand_dims(np.zeros(gray_img.shape), 3)
        #color = np.concatenate([c0,c0,c255], 2)
        #color_img = color_img.astype(np.float32) + color
        #color_img = (color_img / color_img.max()) *255

        cv2.imwrite(filename, gray_img)


def nii2jpg_label(img_path, output_root, img_name=None):
    if img_name is None:
        img_name = (img_path.split('/')[-1]).split('.')[0]
    output_path = os.path.join(output_root, img_name)
    try:
        os.mkdir(output_root)
    except:
        pass
    try:
        os.mkdir(output_path)
    except:
        pass
    img = nib.load(img_path)
    img = (img.get_fdata())[:,:,:]
    #pdb.set_trace()
    img = img*50
    img = img.astype(np.uint8)

    for i in range(img.shape[2]):
        filename = os.path.join(output_path, img_name+'_'+str(i)+'.jpg')
        gray_img = img[:,:,i]
        #color_img = np.expand_dims(gray_img, 3)
        #color_img = np.concatenate([color_img, color_img, color_img], 2)

        # COLOR LABELING
        #c255 = np.expand_dims(np.ones(gray_img.shape)*255, 3)
        #c0 = np.expand_dims(np.zeros(gray_img.shape), 3)
        #color = np.concatenate([c0,c0,c255], 2)
        #color_img = color_img.astype(np.float32) + color
        #color_img = (color_img / color_img.max()) *255

        cv2.imwrite(filename, gray_img)

cnt = 0
for directory in os.listdir(PATH_MASTER):
    cnt += 1
    print(f'{cnt} / {len(os.listdir(PATH_MASTER))}')
    image_t1 = None
    image_label = None
    dir_path = os.path.join(PATH_MASTER, directory)
    for image in os.listdir(dir_path):
        if "_t1.nii" in image:
            image_t1 = image
        if "_seg.nii" in image:
            image_label = image
    if image_t1 is not None and image_label is not None:
        nii2jpg_img(os.path.join(dir_path, image_t1), IMG_OUTPUT_ROOT, img_name=image_t1.replace("_t1.nii", ""))
        nii2jpg_label(os.path.join(dir_path, image_label), LABEL_OUTPUT_ROOT, img_name=image_label.replace("_seg.nii", ""))





# for path in os.listdir(IMG_ROOT):
#     print(path)
#     if path[0] == '.':
#         continue
#     nii2jpg_img(os.path.join(IMG_ROOT,path), IMG_OUTPUT_ROOT)
'''
for path in os.listdir(LABEL_ROOT):
    print(path)
    if path[0] == '.':
        continue
    nii2jpg_label(os.path.join(LABEL_ROOT,path), LABEL_OUTPUT_ROOT)
'''
