# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:12:20 2020
@author: admin
"""
from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, merge, Dense, Activation, Conv2D, Convolution2D, MaxPooling2D, Flatten, Dropout, LeakyReLU, ReLU
from keras.optimizers import Adam
from keras.optimizers import SGD
from numpy.random import randint

from numpy import load
def get_alexnet(): 
	main_input = Input(shape=(256,256,3), dtype='float32', name='main_input') 
# Conv layer 1 output shape (55, 55, 48)
	conv_1 = Convolution2D(nb_filter=48, nb_row=11, nb_col=11, subsample=(4, 4), activation='relu', name='conv_1', init='he_normal', dim_ordering='tf')(main_input) 
	conv_1 = Dropout(0.25)(conv_1)
    
    # Conv layer 2 output shape (27, 27, 128)
	conv_2 = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, subsample=(2, 2), activation='relu', name='conv_2', init='he_normal')(conv_1)
	conv_2 = Dropout(0.25)(conv_2)
    
    # Conv layer 3 output shape (13, 13, 192)
	conv_3 = Convolution2D(nb_filter=192, nb_row=3, nb_col=3, subsample=(2, 2), border_mode='same', activation='relu', name='conv_3', init='he_normal')(conv_2)
	conv_3 = Dropout(0.25)(conv_3)
    
    # Conv layer 4 output shape (13, 13, 192)
	conv_4 = Convolution2D(nb_filter=192, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='conv_4', init='he_normal')(conv_3)
	conv_4 = Dropout(0.25)(conv_4)
    
    # Conv layer 5 output shape (13, 128, 128)
	conv_5 = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same', name='conv_5', init='he_normal')(conv_4)
	conv_5 = Dropout(0.25)(conv_5)
    
    # fully connected layer 1
	flat = Flatten()(conv_5)
	dense_1 = Dense(2048, activation='relu', name='dense_1', init='he_normal')(flat)
	dense_1 = Dropout(0.25)(dense_1)
    
    # fully connected layer 2
	dense_2 = Dense(2048, activation='relu', name='dense_2', init='he_normal')(dense_1)
	dense_2 = Dropout(0.25)(dense_2)
    
    # output
 	# label = Dense(1, activation='leakyRelu', name='label', init='he_normal')(dense_2)
	label = Dense(1, name='label', activation="tanh", init='he_normal')(dense_2)
# 	label = LeakyReLU()(label)
    
    # build CNN model
	model = Model(input=main_input, output=label)
	return model

def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	return [X1, X2]

alexnet = get_alexnet()
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
alexnet.compile(optimizer=sgd, loss='mae', metrics=['mae'])

print(alexnet.summary())

imgs, labs = load_real_samples('trainDataIron.npz')
valimgs, vallabs = load_real_samples('validDataIron.npz')
imgs = np.reshape(imgs, (len(imgs), 256, 256, 3))
valimgs = np.reshape(valimgs, (len(valimgs), 256, 256, 3))



newLabs = np.asarray(labs)
newvalLabs = np.asarray(vallabs)
alexnet.fit(imgs, newLabs, epochs=20, batch_size=32,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet.fit(imgs, newLabs, epochs=80, batch_size=15,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet.fit(imgs, newLabs, epochs=80, batch_size=15,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet.fit(imgs, newLabs, epochs=80, batch_size=15,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet.fit(imgs, newLabs, epochs=80, batch_size=15,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet.fit(imgs, newLabs, epochs=80, batch_size=15,verbose=1, validation_data=(valimgs, newvalLabs))
# alexnet = get_alexnet()
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)

alexnet.compile(optimizer=sgd, loss='mae', metrics=['mae'])
alexnet.fit(imgs, newLabs, epochs=20, batch_size=32,verbose=1, validation_data=(valimgs, newvalLabs))

alexnet.save("alexNet32535.h5")

#0.1254
# for epoch in range(2000):
#     for i in range(7749):
#         idx = randint(0, len(imgs), 15)
#         img = imgs[idx]
#         lab = labs[idx]
#         img = np.reshape(img, (15,128,128,1))
#         oneHot = []
#         for l in lab:
#             t = [0] * 6
#             t[l] += 1
#             oneHot.append(t)
#         oneHot = np.array(oneHot)
#         output = alexnet.train_on_batch(img, oneHot, class_weight="auto")
#         print(epoch, i, output)
    
    
#     if (epoch+1) % 100 == 0:
#         name = 'models/8model_%06d.h5' % (epoch+1)
#         alexnet.save(name)