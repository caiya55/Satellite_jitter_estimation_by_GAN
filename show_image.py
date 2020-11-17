# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 14:10:30 2018

@author: zhang
"""
import tensorflow as tf
from model_TF import D_on_G
from config import get_config
#from data import MNISTDataHandler
from ops import mkdir
from keras.models import load_model
from utils import load_images, load_images_with_C, load__class_images, load__class_images_with_C
import numpy as np
import matplotlib.pyplot as plt
from dataset_process_attitude_farm import jitter_maker_4sin, jitter2D, jitter_with_curve
from PIL import Image
from random import randint
import cv2
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--final_layer", type=int, help="choose the number of final layers", default = 128)
parser.add_argument("--alpha", type=float, help="choose the value of alpha", default = 1)
parser.add_argument("--max_pooling", type=bool, help="choose whether max_pooling is used", default = True)
parser.add_argument("--kernel_size", type=int, help="choose size of the kernel", default = 3)

args = parser.parse_args()

if __name__ == "__main__":
  sess = tf.Session()   
  config = get_config(is_train=True)
  restore = D_on_G(sess, config, "DIRNet", args, is_train=True)
  restore.restore(config.ckpt_dir)  
  bo = 20
  
  # figure 2
  yaogan_x = preprocess('..//..//dataset//yaogan26//for_classification_air.png', 0, 0)
  _,_, output,wrap_yaogan = restore.predict_one(yaogan_x[0], config)        
          
  plt.imshow(output[:,:,0], cmap='gray')
  plt.grid(False)
  plt.axis('off')   
  plt.title('Rstored image')
  cv2.imwrite('results//restored2.png', uint_img(output[:,:,0]))

  plt.show()            
    
  plt.plot(wrap_yaogan[:,0])
  plt.title('Obtained attitude cross-track')
  plt.grid(True)
  plt.xlabel('Lines')
  plt.ylabel('Pixel')
  plt.savefig('results//cross-track-curve3.png')
  plt.show()
  np.save('results//air_jitter.npy',wrap_yaogan)

  # figure 3
  yaogan_x = preprocess('..//..//dataset//yaogan26//for_classification_pale.png', 0, 0)
  _,_, output,wrap_yaogan = restore.predict_one(yaogan_x[0], config)        
          
  plt.imshow(output[:,:,0], cmap='gray')
  plt.grid(False)
  plt.axis('off')   
  plt.title('Rstored image')
#  cv2.imwrite('results//restored1.png', uint_img(output[:,:,0]))

  plt.show()            
    
  plt.plot(wrap_yaogan[:,0])
  plt.title('Obtained attitude cross-track')
  plt.grid(True)
  plt.xlabel('Lines')
  plt.ylabel('Pixel')
  plt.savefig('results//cross-track-curve3.png')
  plt.show()
  np.save('results//pale_jitter.npy',wrap_yaogan)
