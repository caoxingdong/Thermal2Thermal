# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 00:14:27 2021
@author: admin
"""
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
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

iv3 = InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(128,128,3))

x = AveragePooling2D()(iv3.output)

x = Flatten()(x)

x = Dense(512, activation='relu')(x)

x = Dense(512, activation='relu')(x)

label = Dense(41, activation='softmax')(x)

model = Model(iv3.input, label)
model.summary()

for layer in model.layers[:312]:
    layer.trainable = False

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


imgs, labs = load_real_samples('trainData123.npz')

for epoch in range(50):
    for i in range(123):
        idx = randint(0, len(imgs), 15)
        img = imgs[idx]
        lab = labs[idx]
        img= np.stack((img, img, img), 3)
        oneHot = []
        for l in lab:
            t = [0] * 41
            t[l-1] += 1
            oneHot.append(t)
        oneHot = np.array(oneHot)
        output = model.train_on_batch(img, oneHot)
        print(epoch, i, output)
    
    
    if (epoch+1) % 5 == 0:
        name = 'models/iv123_1_%06d.h5' % (epoch+1)
        model.save(name)
        
for layer in model.layers:
    layer.trainable = True

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# imgs, labs = load_real_samples('trainData.npz')

for epoch in range(50):
    for i in range(123):
        idx = randint(0, len(imgs), 15)
        img = imgs[idx]
        lab = labs[idx]
        img= np.stack((img, img, img), 3)
        oneHot = []
        for l in lab:
            t = [0] * 41
            t[l-1] += 1
            oneHot.append(t)
        oneHot = np.array(oneHot)
        output = model.train_on_batch(img, oneHot)
        print(epoch, i, output)
    
    
    if (epoch+1) % 5 == 0:
        name = 'models/iv123_2_%06d.h5' % (epoch+1)
        model.save(name)