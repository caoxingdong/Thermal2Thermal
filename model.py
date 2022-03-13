# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 11:54:49 2020
@author: admin
"""

import os 

import numpy as np
import cv2

from keras.models import load_model

from keras import models

from keras.models import Input
from keras.models import Model

from keras.layers import Concatenate
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ReLU
from keras.layers import Conv2DTranspose
from keras.layers import Add
from keras.layers import Dropout
from keras.applications.inception_v3 import InceptionV3
import random

from keras.optimizers import Adam

from keras.initializers import RandomNormal

from keras import activations

from keras.applications import VGG19
from numpy.random import randint

import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
import keras.backend as K

from PIL import Image

class faceAging():
    
    def __init__(self):
        self.batchSize = 1
        self.image_row = 256
        self.image_col = 256
        self.channel = 3
        self.nClasses = 6
        self.image_shape = (self.image_row, self.image_col, self.channel)
        self.patch_row = self.image_row // 16
        self.patch_col = self.image_col // 16
        self.patch_shape = (self.batchSize, self.patch_row, self.patch_col, 1)       
        
        
        
        self.trainDataFileName = 'trainDataIron.npz'
        self.testDataFileName = 'testDataIron.npz'
        self.t128 = []
        for i in range(self.nClasses):
            arr = [0] * self.nClasses
            arr[i] += 1
            self.t128.append(np.array([[arr for _ in range(256)] for _ in range(256)]))
        
        self.t64 = []
        for i in range(self.nClasses):
            arr = [0] * self.nClasses
            arr[i] += 1
            self.t64.append(np.array([[arr for _ in range(64)] for _ in range(64)]))        
        
        self.tLabel = []
        for i in range(self.nClasses):
            arr = [0] * self.nClasses
            arr[i] += 1
            self.tLabel.append(arr)
        self.iters_per_check = 20   
        self.iterations = 3000000
        self.real_patch_out = np.ones(self.patch_shape, dtype=np.float32)
        self.fake_patch_out = np.zeros(self.patch_shape, dtype=np.float32)              
        
        self.tClassifierName = 'alexNet.h5'
        self.tClassifier = self.degineTClassifier()
        self.myVGG = self.defineMyVGG()
        self.G = self.defineG()
        self.D = self.defineD()
        self.GAN = self.defineGAN()
        
    def defineMyVGG(self):
        # vgg = VGG19(include_top=False, input_shape=(self.image_row, self.image_row, 3))
        # vgg.(train)able = False
        # for layer in vgg.layers:
        #     layer.trainable = False
        # model = Model(inputs=vgg.input, outputs=vgg.layers[5].output)
        # model.trainable = False
        # return model
        vgg = VGG19(include_top=False, input_shape=(self.image_row, self.image_row, 3))
        vgg.trainable = False
        models = []
        for i in [2,5,10,15]:
            model = Model(inputs=vgg.input, outputs=vgg.layers[i].output)
            model.trainable = False
            models.append(model)
        return models
    
    def degineTClassifier(self):
        model = load_model(self.tClassifierName)
        model.trainable = False
        for layer in model.layers:
            layer.trainable = False
        return model        
        
    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):        
     	init = RandomNormal(stddev=0.02)         
     	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)         
     	if batchnorm:
              g = BatchNormalization()(g, training=True)              
     	g = LeakyReLU(alpha=0.2)(g)
     	return g    
    
    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
     	init = RandomNormal(stddev=0.02)
     	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
     	g = BatchNormalization()(g, training=True)
     	if dropout:
     	    g = Dropout(0.5)(g, training=True)
     	g = Concatenate()([g, skip_in])
     	g = Activation('relu')(g)
     	return g     
     
    def defineG(self):
     	init = RandomNormal(stddev=0.02)
     	inputTheraml = Input(shape=(256, 256, 3))
     	targetT = Input(shape=(256, 256, 1))
     	ipt = Concatenate()([inputTheraml, targetT])
     	e1 = self.define_encoder_block(ipt, 64, batchnorm=False)
     	e2 = self.define_encoder_block(e1, 128)
     	e3 = self.define_encoder_block(e2, 256)
     	e4 = self.define_encoder_block(e3, 512)
     	e5 = self.define_encoder_block(e4, 512)
     	e6 = self.define_encoder_block(e5, 512)
     	e7 = self.define_encoder_block(e6, 512)
     	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
     	b = Activation('relu')(b)
     	d1 = self.decoder_block(b, e7, 512)
     	d2 = self.decoder_block(d1, e6, 512)
     	d3 = self.decoder_block(d2, e5, 512)
     	d4 = self.decoder_block(d3, e4, 512, dropout=False)
     	d5 = self.decoder_block(d4, e3, 256, dropout=False)
     	d6 = self.decoder_block(d5, e2, 128, dropout=False)
     	d7 = self.decoder_block(d6, e1, 64, dropout=False)
     	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
     	predictThermal = Activation('tanh')(g)
     	model = Model([inputTheraml, targetT], predictThermal)
     	model.summary()
     	return model     
     
    def defineD(self):
        init = RandomNormal(stddev=0.02)
        thermal = Input(shape=(256, 256, 3))
        # tLabel = Input(shape=(64, 64, self.nClasses))
        
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(thermal) # kernel_initializer=init?
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        # x = Concatenate()([x, tLabel])
        
        x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
        patch_out = Activation('sigmoid')(x)
        
        model = Model(thermal, patch_out)
        opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[1])
        model.summary()
        return model
    
    def defineGAN(self):        
        self.D.trainable = False        
        inputThermal = Input(shape=(256, 256, 3))
        targetT128 = Input(shape=(256, 256, 1))
        # labelT64 = Input(shape=(64, 64, self.nClasses))        
        
        def featureMatchingLoss(y_true, y_pred, models=self.myVGG):                 
            # model.trainable = False
            # y_true = Concatenate()([y_true, y_true, y_true])
            # y_pred = Concatenate()([y_pred, y_pred, y_pred])
            # loss = K.mean(K.abs(model(y_true) - model(y_pred)))
            # return loss  
            loss_vgg19 = 0
            for model in models:
                model.trainable = False
                loss_vgg19 += K.mean(K.abs(model(y_true) - model(y_pred)))
            return loss_vgg19/len(models)
        
        predictedThermal = self.G([inputThermal, targetT128])
        # print(predictedThermal)
        patchOut = self.D(predictedThermal)     
        tPredict = self.tClassifier(predictedThermal)   
        
        model = Model(inputs=[inputThermal, targetT128], outputs=[patchOut, predictedThermal, tPredict])
        model.summary()
        opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
        model.compile(loss=['binary_crossentropy', featureMatchingLoss, "mae"], optimizer=opt, loss_weights=[1, 100, 500])
        return model
    
    # def trainDataLoader(self):
    #     batchSize = self.batchSize
    #     filename = self.trainDataFileName
    #     train_data = np.load(filename, allow_pickle=True)
    #     print("load successfully")
    #     thermalImgs, realT = train_data['arr_0'], train_data['arr_1']
    #     size = len(thermalImgs)
    #     ids = list(range(size))    
    #     batchs = size//batchSize
    #     while True:
    #         np.random.shuffle(ids)
    #         r = np.random.randint(1,self.nClasses)
    #         for i in range(batchs):
    #             ids_this_batch = ids[i*batchSize:(i+1)*batchSize]
    #             thermalImgs_this_batch = [thermalImgs[idx] for idx in ids_this_batch]
    #             realT_this_batch = [realT[idx] for idx in ids_this_batch]
    #             realT128 = []
    #             realT64 = []
    #             fakeT128 = []
    #             fakeT64 = []
    #             realTLabel = []
    #             fakeTLabel = []
    #             for t in realT_this_batch:
    #                 realT128.append(self.t128[t])
    #                 realT64.append(self.t64[t])
    #                 realTLabel.append(self.tLabel[t])
    #                 ft = (t + r) % 6
    #                 fakeT128.append(self.t128[ft])
    #                 fakeT64.append(self.t64[ft])
    #                 fakeTLabel.append(self.tLabel[ft])
    #             thermalImgs_this_batch = np.array(np.reshape(thermalImgs_this_batch, (batchSize, self.image_row, self.image_col, self.channel)))           
    #             realT128 = np.array(np.reshape(realT128, (batchSize, 128, 128, self.nClasses)))                
    #             fakeT128 = np.array(np.reshape(fakeT128, (batchSize, 128, 128, self.nClasses)))                               
    #             realT64 = np.array(np.reshape(realT64, (batchSize, 64, 64, self.nClasses)))               
    #             fakeT64 = np.array(np.reshape(fakeT64, (batchSize, 64, 64, self.nClasses)))
    #             realTLabel = np.array(np.reshape(realTLabel, (batchSize, self.nClasses)))
    #             fakeTLabel = np.array(np.reshape(fakeTLabel, (batchSize, self.nClasses)))
    #             yield thermalImgs_this_batch, realT128, fakeT128, realT64, fakeT64, realTLabel, fakeTLabel
    
    def train(self):
        batchSize = self.batchSize
        filename = self.trainDataFileName
        testname = self.testDataFileName
        train_data = np.load(filename, allow_pickle=True)
        test_data = np.load(testname, allow_pickle=True)
        thermalTrainImgs, realT = train_data['arr_0'], train_data['arr_1']
        real64s1 = []
        for t in realT:
            real64s1.append(np.array([[t]*256]*256))
        thermalTestImgs, testRts, testIds, testSessions = test_data['arr_0'], test_data['arr_1'], test_data['arr_2'], test_data['arr_3']
        real64s2 = []
        for t in testRts:
            real64s2.append(np.array([[t]*256]*256))
        # size = len(thermalTrainImgs)
        # ids = list(range(size))    
        # batchs = size//batchSize
        # ts = [32.25, 32.75, 33.25, 33.75, 34.25, 34.75]
        t128 = [-0.47, -0.27, -0.07, 0.12, 0.32, 0.52]
        
        self.t128 = []
        for i in range(6):
            self.t128.append(np.array([[t128[i]]*256]*256))
        
        for iter_idx in range(1, self.iterations+1):
            
            
            r = np.random.randint(6)
            ids_this_batch = randint(0, len(thermalTrainImgs), batchSize)
            thermalImgs_this_batch = [thermalTrainImgs[idx] for idx in ids_this_batch]
            realT_this_batch = [realT[idx] for idx in ids_this_batch]
            realT128 = []
            # realT64 = []
            fakeT128 = []
            # fakeT64 = []
            realTLabel = []
            fakeTLabel = []
            for t in realT_this_batch:
                realT128.append([[t]*256]*256)
                # realT64.append(self.t64[t])
                realTLabel.append(t)
                # ft = (t + r) % 6
                ft = random.randrange(-70,90,1)/100
                fakeT128.append([[ft]*256]*256)
                # fakeT64.append(self.t64[ft])
                fakeTLabel.append(ft)
            thermalImgs = np.array(np.reshape(thermalImgs_this_batch, (batchSize, self.image_row, self.image_col, self.channel)))           
            realT128 = np.array(np.reshape(realT128, (batchSize, 256, 256, 1)))                
            fakeT128 = np.array(np.reshape(fakeT128, (batchSize, 256, 256, 1)))                               
            # realT64 = np.array(np.reshape(realT64, (batchSize, 64, 64, self.nClasses)))               
            # fakeT64 = np.array(np.reshape(fakeT64, (batchSize, 64, 64, self.nClasses)))
            realTLabel = np.array(np.reshape(realTLabel, (batchSize, 1)))
            fakeTLabel = np.array(np.reshape(fakeTLabel, (batchSize, 1)))
            
            # thermalImgs, realT128, fakeT128, realT64, fakeT64, realTLabel, fakeTLabel = next(self.trainDataLoader())
            generatedThermal = self.G.predict([thermalImgs, realT128])
            self.D.trainable = True
            self.G.trainable = False
            self.D.train_on_batch([thermalImgs], self.real_patch_out)            
            self.D.train_on_batch([generatedThermal], self.fake_patch_out)
            # self.D.train_on_batch([thermalImgs, realT64], self.real_patch_out)            
            # self.D.train_on_batch([generatedThermal, realT64], self.fake_patch_out)
            # self.D.train_on_batch([thermalImgs, realT64], self.real_patch_out)            
            # self.D.train_on_batch([thermalImgs, fakeT64], self.fake_patch_out)  
            self.D.trainable = False
            self.G.trainable = True
            # res1 = (self.GAN.train_on_batch([thermalImgs, realT128], [self.real_patch_out, thermalImgs, realTLabel]))
            res2 = (self.GAN.train_on_batch([thermalImgs, fakeT128], [self.real_patch_out, thermalImgs, fakeTLabel]))
            if iter_idx % 10 == 0:
                print(iter_idx,ft, res2)
            # self.GAN.train_on_batch([thermalImgs, realT128, realT64], [self.real_patch_out, thermalImgs, realTLabel])
            # self.GAN.train_on_batch([thermalImgs, fakeT128, fakeT64], [self.real_patch_out, thermalImgs, fakeTLabel])
            
            if iter_idx % 5000 == 0:
                # print(res1, res2)
                os.mkdir("testImgs28/%d" %(iter_idx))
                # self.save(iter_idx)
                for ii in range(0, len(thermalTestImgs), 1):
                
                    if ii % 100 == 0:
                        print(ii)
                    ids_this_batch = [ii]
                    thermalImgs_this_batch = [thermalTestImgs[idx] for idx in ids_this_batch]
                    thermalImgs = np.array(np.reshape(thermalImgs_this_batch, (batchSize, self.image_row, self.image_col, self.channel)))           
                    generatedThermalR = self.G.predict([thermalImgs, realT128])
                    generatedThermalF = self.G.predict([thermalImgs, fakeT128])
                    
                    thermalImgs = np.reshape(thermalImgs, (256,256,3))
                    thermalImgs += 1
                    thermalImgs *= 127.5
                    thermalImgs = thermalImgs.astype(np.uint8)
                    thermalImgs = Image.fromarray(thermalImgs)
                    thermalImgs.save('testImgs28/' +str(iter_idx) + "/"+ str(ii+1) + 'R_' + str(testRts[ii])[:5] + '_' + str(testIds[ii]) +  '_S' +str(testSessions[ii]) +'.jpg')
                    
                    
                    for r in range(6):
                        fakeT128 = []
                        for t in realT_this_batch:
                            fakeT128.append(self.t128[r])
                        fakeT128 = np.array(np.reshape(fakeT128, (batchSize, 256, 256, 1)))  
                        thermalImgs = np.array(np.reshape(thermalImgs_this_batch, (batchSize, self.image_row, self.image_col, self.channel)))      
                        generatedThermalR = self.G.predict([thermalImgs, fakeT128])
                        
                        generatedThermalR = np.reshape(generatedThermalR, (256,256,3))                
                        generatedThermalR += 1
                        generatedThermalR *= 127.5
                        generatedThermalR = generatedThermalR.astype(np.uint8)
                        generatedThermalR = Image.fromarray(generatedThermalR)
                        generatedThermalR.save('testImgs28/'  +str(iter_idx) +'/'+ str(ii+1) + 'F_' + str(r) + '_' + str(testIds[ii]) + '_S' +str(testSessions[ii]) +'.jpg')
                    
                # import numpy as np
                from keras.models import load_model
                
                
                def get():
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
                    
                    testImages, testLabels = [], []
                    
                    filefolder = "testImgs28/%d/" %iter_idx
                    
                    files = listdir(filefolder)
                    for i in range(len(files)):
                        f = files[i]
                        if i % 1000 == 0:
                            print(i)
                        t = float(f.split('_')[1])
                        idx = int(f.split('_')[2])
                                  
                        if 'R' in f or idx <= 30:
                            continue
                        img = imageio.imread(filefolder + f)
                        img = img.astype(np.float32)
                        img /= 127.5
                        img -= 1  
                        testImages.append(img)
                        testLabels.append(t)
                    
                    testImages = asarray(testImages)
                    testLabels = asarray(testLabels)
                    return testImages, testLabels
                
                # test_data = np.load("testData4.npz", allow_pickle=True)
                imgs, labels = get()
                model = load_model("alexNet.h5")
                # matrix = [[0] * 6 for _ in range(6)]
                # for i in range(len(imgs)):
                #     img = imgs[i]
                #     img = np.reshape(img, (1, 128, 128, 1))
                #     output = model.predict(img)
                #     output = output[0].tolist()
                #     idx = output.index(max(output))
                #     matrix[idx][labels[i]] += 1
                imgs = np.reshape(imgs, (len(imgs), 256, 256, 3))
                labels = np.asarray(labels)
                score = model.predict(imgs) 
                res = [[] for _ in range(6)]
                for i, s in enumerate(score):
                    res[int(labels[i])].append(s)
                # for i in range(len(imgs)):
                #     img = imgs[i]
                #     img += 1
                #     img /= 2
                #     img *= 255
                #     mx = np.amax(img[:30][:])
                #     matrix[labels[i]][getLaeb(mx)] += 1
                # print(matrix)
                arr = [-0.47, -0.27, -0.07, 0.12, 0.32, 0.52]
                for i in range(6):
                    k = [abs(arr[i]-a) for a in res[i]]
                    print(i, sum(k)/len(k))
    
    def save(self, iter_idx):
        def freeze(model):
            for layer in model.layers:
                layer.trainable = False
            
            if isinstance(layer, models.Model):
                freeze(layer)
                
        G = self.G    
        freeze(G)
        
        G.save("models2/" + str(iter_idx) + 'G_model')
        

    
model = faceAging()
model.train()