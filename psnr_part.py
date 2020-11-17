# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:17:06 2018

@author: zhang
"""

import numpy
import math
import tensorflow as tf
from model_TF import D_on_G
from config import get_config
#from data import MNISTDataHandler
from ops import mkdir
from keras.models import load_model
from utils import load_images, load_images_with_C, load__class_images, load__class_images_with_C
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randint
import cv2
import argparse
import os

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


cla = 'bridge'
#  for cla in classes:
img_dir = '..\\..\\dataset\\image_deform\\' + cla + '\\'
  
data = load__class_images_with_C(img_dir, 200, istrain = False) # load the images from different classes
#  data = load_images('..\\..\\dataset\\image_deform\\',100,istrain = False)# load the images from different classes
y_test, x_test, z_test = data['B'], data['A'], data['C']
x_test = x_test[:,:,:,np.newaxis]
y_test = y_test[:,:,:,np.newaxis]
index = 177
_,_, y_result, wrap_test = restore.predict_one(x_test[index], config)
plt.imshow(x_test[index,:,:,0], cmap='gray')
plt.grid(False)
plt.axis('off')   
plt.title('Rstored image')
plt.show()


raw = x_test[index,:,:,0]
ground = y_test[index,:,:,0]
correct = y_result[:,:,0]

w = 10
lin = int(256/w)
diff  = np.zeros([lin,lin])
for i in range(lin):
    for j in range(lin):
        part_r = raw[i:i+w, j:j+w]
        part_g = ground[i:i+w, j:j+w]
        part_c = correct[i:i+w, j:j+w]
        part_psnr = psnr(part_c,part_g)
        part_b_psnr = psnr(part_r,part_g)
        diff[i,j] = part_psnr-part_b_psnr
    
