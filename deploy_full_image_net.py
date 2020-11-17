# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:48:18 2018

@author: zhang
"""

import tensorflow as tf
from model_TF import D_on_G
from config import get_config
from ops import mkdir
from utils import load_images, load__class_images,load__class_images_with_C
from utils import load__attitude_images
from utils import load_jitter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.interpolate import interp2d

import numpy as np

n_images = 200
def mse(a, b):
    return ((a - b)**2).mean()

def list_mean(a):
    return sum(a)/len(a)
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
#  main()
      sess = tf.Session()
      config = get_config(is_train=False)
#      tf.get_variable_scope().reuse_variables()
      restore = D_on_G(sess, config, "DIRNet", is_train=True)
      restore.restore(config.ckpt_dir)
      data = load__class_images('..\\..\\dataset\\image_deform\\yaogan26\\',n_images,istrain = False)# load the images from different classes

      y_train, x_train = data['B'], data['A']
      x_train = x_train[:,:,:,np.newaxis] 
      y_train = y_train[:,:,:,np.newaxis] 
      error_out = []
      losses = []
      for i in range(0, x_train.shape[0]-5):
          batch_x = x_train[i:i + config.batch_size]
          batch_y = y_train[i:i + config.batch_size] 
          loss, _ = restore.predict(batch_x, batch_y)

          losses.append(loss)
      value = np.mean(losses)
      print('mean error is', value)      
#    
      eout = np.asarray(losses)
      sort = np.argsort(eout)
      index = sort[-1]
      print('eout is', eout[index])
      batch_y = y_train[index:index+5]
      batch_x = x_train[index:index+5]
      loss, output = restore.predict(batch_x, batch_y)
      raw = batch_x[0,:,:,0]
      result = output[:,:,0]
      truth = batch_y[0,:,:,0]
      plt.imshow(raw,cmap='gray')
      plt.grid(False)
      plt.axis('off')
      plt.title('raw_result')
      plt.show()   
         
      plt.imshow(result,cmap='gray')
      plt.grid(False)
      plt.axis('off')
      plt.title('Result')
      plt.show()            

      plt.imshow(truth,cmap='gray')
      plt.grid(False)
      plt.axis('off')
      plt.title('GroundTruth')
      plt.show()            
