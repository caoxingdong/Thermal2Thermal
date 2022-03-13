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
def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	return [X1, X2]

res1 = []
res2 = []
res3 = []
res4 = []

for s in ['123','234','124','134']:

    model = load_model('models/iv%s.h5' %s)
    imgs, labels = load_real_samples('validData%s.npz' %s)
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
    print(cnt)
    c = (cnt[0] / len(imgs))
    a = (1 - bisect.bisect(trueArr, falseArr[int(len(falseArr) * 0.99)]) / len(trueArr))
    b = (1 - bisect.bisect(trueArr, falseArr[int(len(falseArr) * 0.999)]) / len(trueArr))
    res1.append(cnt)
    print(a)
    print(b)
    print(c)
    res2.append(a)
    res3.append(b)
    res4.append(c)
print(res1)
print(res2)
print(res3)
print(res4)