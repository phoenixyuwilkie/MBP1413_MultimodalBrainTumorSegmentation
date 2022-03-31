# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:43:52 2022

@author: felix, phoenix, katie
"""

import os
import numpy as np
import config
import nibabel as nib

def nifti2array(file, modality):
    img = nib.load(file)
    img = img.get_fdata()
    img = img/img.max()
    # img = img.astype(np.uint8)
    return img

def choose_modality(path, modalities, save_checkpoint = None):

    files = os.listdir(path)
    data = []
    for modality in modalities:
        file = [s for s in files if config.FORMAT[modality] in s][0]
        data.append(nifti2array(os.path.join(path, file), modality))

    return data
