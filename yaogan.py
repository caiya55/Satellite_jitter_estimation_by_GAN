# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:26:18 2018

@author: zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:30:32 2018

@author: zhang
"""

import tensorflow as tf
from model_TF import D_on_G
from PIL import Image
from config import get_config
from ops import mkdir
from utils import load_images
from utils import load__attitude_images
from utils import load_jitter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.io as scio  
import numpy as np
import scipy.misc  
from scipy.interpolate import interp2d
from scipy.misc import imresize
import cv2
n_images = 20
def load_image(path):
    img = Image.open(path).convert('L')
    return img

def preprocess(path, x_b, y_b):
    img = load_image(path)
#    cv2.imres
    img = np.array(img) 
#    img = np.rot90(img,2)
#    img = img / 255
    img = img[x_b:x_b+256, y_b:y_b+256]
    img = (img - 127.5) / 127.5
    num = 5
    out = np.zeros([5,256,256,1])
    for i in range(num):
        out[i,:,:,0] = img
    return out
def preprocess_rotate(path):
    img = load_image(path)
    img = np.array(img) 
    img = imresize(img, 0.5)    
    img_raw = img[0:256,100:256+100]
    img = (img_raw - 127.5) / 127.5
    num = 5
    out = np.zeros([5,256,256,1])
    for i in range(num):
        out[i,:,:,0] = img
    return out,img_raw

def compensate(img, jix):
    height = img.shape[0]
    width = img.shape[1]
    x = np.arange(height).astype(float)
    y = np.arange(width).astype(float)
    f = interp2d(x, y, img, kind='linear')
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        out_tmp = f(y+jix[index], x[index]).T
        img_out[index] = out_tmp
    return img_out
if __name__ == "__main__":
      sess = tf.Session()
#      tf.get_variable_scope().reuse_variables()
      config = get_config(is_train=False)  
      restore = D_on_G(sess, config, "DIRNet", is_train=True)
      restore.restore(config.ckpt_dir)
      
      # F9 command area
      yaogan_x = preprocess('..//..//dataset//yaogan26//for_classification.png', 0, 0)
#      yaogan_x, img_raw = preprocess_rotate('..//dataset//yaogan26//airport.png')
#      plt.imshow(img_raw,cmap='gray')
      loss, output = restore.predict_one(yaogan_x[0], config)
      kk = 0
      
      plt.imshow(output[:,:,0], cmap='gray')
      plt.grid(False)
      plt.axis('off')   
      plt.title('Result')
      plt.show()            

      plt.imshow(yaogan_x[0,:,:,0],cmap='gray')
      plt.grid(False)
      plt.axis('off')
      plt.title('GroundTruth')
      plt.show()            

#      wrap_0 = output[kk,:,0]
#      jit_result_raw = wrap_0 - np.mean(wrap_0)
#      jit_result = jit_result_raw
#    # show the result
#      plt.plot(range(jit_result.shape[0]), jit_result)
#      plt.title('Detected jitter result')
#      plt.xlabel('Lines')
#      plt.ylabel('Pixels')
#      plt.show()
#
#      raw = yaogan_x[0,:,:,0]
#      
#      plt.imshow(raw,cmap='gray')
#      plt.grid(False)
#      plt.title('Raw')
#      plt.axis('off')
#      plt.show()
#      out = compensate(raw, -jit_result)
#      plt.imshow(out,cmap='gray')
#      plt.grid(False)
#      plt.axis('off')
#      plt.title('Restored')
#      plt.show()      
