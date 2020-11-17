import tensorflow as tf
import os
#import skimage.io
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

def conv2d(x, name, dim, k, s, p, bn, af, is_train):
  with tf.variable_scope(name):
    w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
      initializer=tf.truncated_normal_initializer(stddev=0.01))
    x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

    if bn:
      x = batch_norm(x, "bn", is_train=is_train)
    else :
      b = tf.get_variable('biases', [dim],
        initializer=tf.constant_initializer(0.))
      x += b

    if af:
      x = af(x)

  return x

def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
  return tf.contrib.layers.batch_norm(x, 
    decay=momentum,
    updates_collections=None,
    epsilon=epsilon,
    scale=True,
    is_training=is_train, 
    scope=name)

def ncc(x, y):
  mean_x = tf.reduce_mean(x, [1,2,3], keep_dims=True)
  mean_y = tf.reduce_mean(y, [1,2,3], keep_dims=True)
  mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
  mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
  stddev_x = tf.reduce_sum(tf.sqrt(
    mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
  stddev_y = tf.reduce_sum(tf.sqrt(
    mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
  return tf.abs(tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y)))

def mse(x, y):
  return tf.reduce_mean(tf.square(x - y))
def mse_without_max(x,y):
    return tf.reduce_mean(tf.square(x-y))/tf.reduce_max(x)

def mkdir(dir_path):
  try :
    os.makedirs(dir_path)
  except: pass 

def save_image_with_scale(path, arrx,arry,arrz):
  arrz = np.clip(arrz, 0., 1.)
  arrz = arrz * 255.
  arrz = arrz.astype(np.uint8)
  arrx = (arrx*255).astype(np.uint8)
  arry = (arry*255).astype(np.uint8)
  arr = np.concatenate((arry, arrx, arrz), axis=1)
  plt.imshow(arr,cmap='gray')
  plt.show()
#  skimage.io.imsave(path, arr)


def show_image_difference(img1, img2):
   dim = abs(img1 - img2)
   dim = np.clip(dim, 0., 1.)
   dim = dim*255
   dim = dim.astype(np.uint8)
   sns.heatmap(dim)
#   plt.imshow(dim, cmap='gray')
#   plt.show()