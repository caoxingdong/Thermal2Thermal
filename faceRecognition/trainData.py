# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:15:42 2020
@author: admin
"""

import random

from numpy import asarray
from numpy import savez_compressed

import os
import imageio
import cv2

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from os import listdir


folder = "E:/xdcao/visual2thermal/database/7 Carl Database/Carl Database/faceAging model/testImgs26/80000/"

files = listdir(folder)


for v in range(1, 5):
    trainImages, trainLabels = [], []
    validImages, validLabels = [], []
    testImages, testLabels = [], []
    want = [1,2,3,4]
    want.pop(want.index(v))
    for i, f in enumerate(files):
        if i % 1000 == 1:
            print(i)
        # session = ((int(f[:f.index('_')-1]) - 1)  % 60) // 15 + 1
        session = int(f[f.index('S')+1])
        part3 = f.split('_')[2]
        label = int(part3)
        img = imageio.imread(folder + f)
        img = img.astype(np.float32)
        img /= 127.5
        img -= 1
        if session in want and 'R' in f:
            trainImages.append(img)
            trainLabels.append(label)
        if session not in want and 'R' in f:
            validImages.append(img)
            validLabels.append(label)
        if session not in want and 'F' in f and int(f.split('_')[1]) >= 2:
            testImages.append(img)
            testLabels.append(label)
            
    s = ''.join(list(map(str, want)))
    trainImages = asarray(trainImages)
    trainLabels = asarray(trainLabels)
    print('Loaded: ', trainImages.shape, trainLabels.shape)
    # save as compressed numpy array
    filename = 'trainData%s.npz' %s
    # savez_compressed(filename, trainImages, trainLabels)
    print('Saved dataset: ', filename)
    
    validImages = asarray(validImages)
    validLabels = asarray(validLabels)
    print('Loaded: ', validImages.shape, validLabels.shape)
    filename = 'validData%s.npz' %s
    print('Saved dataset: ', filename)
    
    
    testImages = asarray(testImages)
    testLabels = asarray(testLabels)
    print('Loaded: ', testImages.shape, testLabels.shape)
    filename = 'testData%s.npz' %s
    savez_compressed(filename, testImages, testLabels)
    print('Saved dataset: ', filename)