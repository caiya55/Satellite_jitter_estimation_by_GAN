# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:53:27 2018

@author: zhang
"""

import tensorflow as tf

import numpy as np

from WarpST_one import WarpST_one 
from utils import load_images, load_images_with_C, load__class_images_with_C
import matplotlib.pyplot as plt
from keras.layers.core import Dropout, Dense, Flatten, Lambda

inputs = tf.placeholder(tf.float32, shape=[5,256,256,1])
wraps = tf.placeholder(tf.float32, shape=[5,256,256,2])
wraps_inputs=np.ones([5,256,256,2])*-0.5

data = load__class_images_with_C('..\\..\\dataset\\image_deform\\freeway\\',5,istrain = True) # load the images from different classes
y_train, x_train = data['C'], data['A']
x_train = x_train[:,:,:,np.newaxis]

y_batch = np.expand_dims(y_train, 2)
y_batch = np.tile(y_batch, [1,1,256,1])

outputs = Lambda(WarpST_one, arguments={'inputs':inputs, 'name':'cha'})(wraps)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(outputs, feed_dict={inputs:x_train, wraps:-y_batch})


plt.imshow(output[0,:,:,0],cmap='gray')