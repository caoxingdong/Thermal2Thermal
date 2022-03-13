# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:31:10 2020
@author: admin
"""

from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
import numpy as np
import bisect

arr = ['123','234','124','134']
models = []
dataset = []
for i in range(4):
    models.append(load_model('models/mn%s.h5' %arr[i]))
    dataset.append(load('testData%s.npz' %arr[i]))



def sub(sub):
    def load_real_samples(i):
    	data = dataset[i]
    	X1, X2 = data['arr_0'], data['arr_1']
    	x1 = []
    	x2 = []
    	for i in range(len(X1)):
            if i % 4 == sub:
                x1.append(X1[i])
                x2.append(X2[i])
    	X1, X2 = x1, x2
    	X1 = np.array(X1)        
    	X2 = np.array(X2)
    	return [X1, X2]
    
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    
    for i, s in enumerate(['123','234','124','134']):
    
        model = models[i]
        imgs, labels = load_real_samples(i)
        # imgs = imgs[2::4]
        # labels = labels[2::4]
        # imgs = np.stack((imgs, imgs, imgs), 3)
        
        cnt = [0] * 41
        trueArr = []
        falseArr = []
        
        res = model.predict(imgs)
        for i in range(len(imgs)):
            r = res[i]
            r = [[v, j+1] for j, v in enumerate(r)]
            r.sort(reverse=True)
            for j in range(41):
                if r[j][1] == labels[i]:
                    trueArr.append(r[j][0])
                    cnt[j] += 1
                else:
                    falseArr.append(r[j][0])
        trueArr.sort()
        falseArr.sort()
        # print(cnt)
        c = (cnt[0] / len(imgs))
        a = (1 - bisect.bisect(trueArr, falseArr[int(len(falseArr) * 0.99)]) / len(trueArr))
        b = (1 - bisect.bisect(trueArr, falseArr[int(len(falseArr) * 0.999)]) / len(trueArr))
        res1.append(cnt)
        # print(a)
        # print(b)
        # print(c)
        res2.append(a)
        res3.append(b)
        res4.append(c)
    return res2, res3,  res4
    # print(res1)
    # print(res2)
    # print(res3)
    # print(res4)
res1 = []
res2 = []
res3 = []
for i in range(4):
    print(i)
    x, y ,z = sub(i)
    print(x,y,z)
    res1.append(x)
    res2.append(y)
    res3.append(z)
print(res1, res2, res3)