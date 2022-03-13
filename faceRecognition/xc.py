# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:51:27 2021
@author: admin
"""

from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from numpy.random import randint
from numpy import load
import numpy as np
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	return [X1, X2]

for s in ['134','234','123','124']:

    mn = Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(256,256,3))
    
    # print(len(xc.layers))
    
    x = AveragePooling2D()(mn.output)
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    
    x = Dense(512, activation='relu')(x)
    
    label = Dense(41, activation='softmax')(x)
    
    model = Model(mn.input, label)
    model.summary()
    
    #mnbineNet: 89
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    sgd = SGD(lr=0.001, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    imgs, labs = load_real_samples('trainData%s.npz' %s)
    
    for epoch in range(50):
        for i in range(len(imgs)//15):
            idx = randint(0, len(imgs), 15)
            img = imgs[idx]
            lab = labs[idx]
            oneHot = []
            for l in lab:
                t = [0] * 41
                t[l-1] += 1
                oneHot.append(t)
            oneHot = np.array(oneHot)
            output = model.train_on_batch(img, oneHot)
            print(epoch, i, output)
            
    for layer in model.layers:
        layer.trainable = True
    
    sgd = SGD(lr=0.0002, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    

    
    for epoch in range(50):
        for i in range(len(imgs)//15):
            idx = randint(0, len(imgs), 15)
            img = imgs[idx]
            lab = labs[idx]
            oneHot = []
            for l in lab:
                t = [0] * 41
                t[l-1] += 1
                oneHot.append(t)
            oneHot = np.array(oneHot)
            output = model.train_on_batch(img, oneHot)
            print(epoch, i, output)
    name = 'models/xc%s.h5' %s
    model.save(name)