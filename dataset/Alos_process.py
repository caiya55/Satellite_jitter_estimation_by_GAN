# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:50:56 2018

@author: zhang
"""


import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from random import random
from numpy.random import random_sample
from numpy import sin

def load_image(path):
    img = Image.open(path).convert('L')
    return img

def preprocess(img):
    img = np.array(img) 
    img = img / 255
    return img
def jitter():
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    ra = random()*0.9 + 0.1
    f = [f*ra for f in f]
    #pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)
    
    amp = [1, 0.3, 0.1, 0.05]
    amp = [amp*random()*3 for amp in amp]
#    fy = 0.2
#    b = 1
    jix_1  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
    #jix = b*sin(fy*y+b)
    jix = jix_1
    jix[0:5] = 0
    jix[-5:] = 0
    return jix
#img_gray = load_image('save.jpg')
#
#img = preprocess(img_gray)
#
#img
img_gray = load_image('yaogan26//f1.jpg')
img = preprocess(img_gray)
width = img.shape[0]
x = np.arange(width).astype(float)
y = np.arange(width).astype(float)

f = interp2d(x, y, img, kind='linear')
jix = jitter()
x_i = x 
y_i = y + jix

# make a loop
img_out = np.zeros(img.shape)
for index in range(img.shape[0]):
    out_tmp = f(y+jix[index], x[index]).T
    img_out[index] = out_tmp
img_out = img_out * 255.
img_out = img_out.astype(np.uint8)

plt.imshow(img_out,cmap='gray')
plt.grid(False)

plt.show()

plt.plot(range(len(jix)), jix)
plt.show()

save_path_A = 'yaogan26//f2.jpg'
jitter_path = 'yaogan26//f2.jpg.npy'
Im = Image.fromarray(img_out)
Im.save(save_path_A)      
np.save(jitter_path,jix)









