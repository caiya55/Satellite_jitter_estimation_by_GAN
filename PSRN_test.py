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


def obtain_classes(path):
#    path = 'images/'
    sub = True
    classes = []
    for root, dirs, files in os.walk(path):
        if sub:
            for cla in dirs:
                classes.append(cla)
        sub = False        
    return classes

def load_image(path):
    img = Image.open(path).convert('L')
    return img

def preprocess(path, x_b, y_b):
    img = load_image(path)
#    cv2.imres
    img = np.array(img) 
#    if randint(0,1)==0:
#        img = np.rot90(img,2)
#        img = img / 255
    img = img[x_b:x_b+256, y_b:y_b+256]
    img = (img - 127.5) / 127.5
    num = 5
    out = np.zeros([5,256,256,1])
    for i in range(num):
        out[i,:,:,0] = img
    return out

def uint_img(img):
    return (img*127.5+127.5).astype(np.uint8)
def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def ncc(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


parser = argparse.ArgumentParser()
parser.add_argument("--final_layer", type=int, help="choose the number of final layers", default = 128)
parser.add_argument("--alpha", type=float, help="choose the value of alpha", default = 1)
parser.add_argument("--max_pooling", type=bool, help="choose whether max_pooling is used", default = True)
parser.add_argument("--kernel_size", type=int, help="choose size of the kernel", default = 1)
args = parser.parse_args()


if __name__ == "__main__":
  sess = tf.Session()   
  config = get_config(is_train=True)
  restore = D_on_G(sess, config, "DIRNet", args, is_train=True)
  restore.restore(config.ckpt_dir)  
  bo = 20
  classes = obtain_classes('..\\..\\dataset\\image_deform\\')
  
#  data = load_images('..\\..\\dataset\\image_deform\\',600,istrain = True) # load the images from different classes
#  y_train, x_train, z_train = data['B'], data['A'], data['C']
#  x_train = x_train[:,:,:,np.newaxis]
#  y_train = y_train[:,:,:,np.newaxis] 
#  z_train = z_train[:,:,:,np.newaxis]
  # load the test dataset
  class_mean_psnr = []
  class_mean_psnr_base = []
#  cla = 'bridge'
  for cla in classes:
      img_dir = '..\\..\\dataset\\image_deform\\' + cla + '\\'
      data = load__class_images_with_C(img_dir, 200, istrain = False) # load the images from different classes
    #  data = load_images('..\\..\\dataset\\image_deform\\',100,istrain = False)# load the images from different classes
      y_test, x_test, z_test = data['B'], data['A'], data['C']
      x_test = x_test[:,:,:,np.newaxis]
      y_test = y_test[:,:,:,np.newaxis]
    #  z_test = z_test[:,:,:,np.newaxis]
      batch_size = config.batch_size
      kernel_past = []
      validation_loss = []
      loss_test_all = []
      psnr_all = []
      psnr_base_all = []
      ed = 5
      for index in range(x_test.shape[0]):
        _,_, y_result, wrap_test = restore.predict_one(x_test[index], config)
        psnr_i = ncc(y_result[ed:-ed,ed:-ed,0], y_test[index,ed:-ed,ed:-ed,0])
        psnr_base_i = ncc(x_test[index,ed:-ed,:,0], y_test[index,ed:-ed,:,0])
        psnr_all.append(psnr_i)
        psnr_base_all.append(psnr_base_i)
      print(cla, ' mean psnr is ', np.mean(psnr_all))
      print(cla, ' mean base psnr is ', np.mean(psnr_base_all))
      class_mean_psnr.append(np.mean(psnr_all))
      class_mean_psnr_base.append(np.mean(psnr_base_all))
  # show the immage
  
  
#### -----------------------output the image
#  index = 176
#  _,_, y_result, wrap_test = restore.predict_one(x_test[index], config)
#  psnr(y_result[:,:,0], y_test[index,:,:,0])
#  psnr(x_test[index,:,:,0], y_test[index,:,:,0])
#  plt.imshow(y_result[:,:,0], cmap='gray')
#  plt.grid(False)
#  plt.axis('off')   
#  plt.title('Rstored image')
#  plt.show()
#  cv2.imwrite('results//'+cla+'1.png', uint_img(y_result[:,:,0]))
#
#  plt.imshow(x_test[index,:,:,0], cmap='gray')
#  plt.grid(False)
#  plt.axis('off')   
#  plt.title('Rstored image')
#  plt.show()
#  cv2.imwrite('results//'+cla+'2.png', uint_img(x_test[index,:,:,0]))
#
#  plt.imshow(y_test[index,:,:,0], cmap='gray')
#  plt.grid(False)
#  plt.axis('off')   
#  plt.title('Rstored image')
#  plt.show()
#  cv2.imwrite('results//'+cla+'3.png', uint_img(y_test[index,:,:,0]))

#  
#  plt.plot(psnr_all)
#  plt.grid(True)
#  plt.show()
np.save('ncc_all.npy', class_mean_psnr)
np.save('ncc_all_baseline.npy', class_mean_psnr_base)
np.save('classes_names.npy', classes)