# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 00:33:17 2018

@author: zhang
"""

import tensorflow as tf
from model_TF import D_on_G
from config import get_config
#from data import MNISTDataHandler
from ops import mkdir
from keras.models import load_model
from utils import load_images, load_images_with_C
import numpy as np
n_images = 5

def write_log(callback, names, logs, batch_no):  
    for name, value in zip(names, logs):  
        summary = tf.Summary()  
        summary_value = summary.value.add()  
        summary_value.simple_value = value  
        summary_value.tag = name  
        callback.writer.add_summary(summary, batch_no)  
        callback.writer.flush()  
        
if __name__ == "__main__":
#  main()
  sess = tf.Session()
  config = get_config(is_train=True)
#  tf.get_variable_scope().reuse_variables()
  mkdir(config.tmp_dir) 
  mkdir(config.ckpt_dir)

  restore = D_on_G(sess, config, "DIRNet", is_train=True)
#  restore.restore(config.ckpt_dir)
#  dh = MNISTDataHandler("MNIST_data", is_train=True)
  data = load_images('..\\..\\dataset\\image_deform\\',n_images,istrain = True) # load the images from different classes
  y_train, x_train = data['B'], data['A']
  x_train = x_train[:,:,:,np.newaxis]
  y_train = y_train[:,:,:,np.newaxis]
  # load the test dataset
  data = load_images('..\\..\\dataset\\image_deform\\',n_images,istrain = False) # load the images from different classes
  y_test, x_test = data['B'], data['A']
  x_test = x_test[:,:,:,np.newaxis]
  y_test = y_test[:,:,:,np.newaxis]

  train_writer = tf.summary.FileWriter('cnn/train', sess.graph)
  test_writer = tf.summary.FileWriter('cnn/test')
  batch_size = config.batch_size
#  merged_summary = tf.summary.merge_all() 
#  all_writer.add_graph(sess.graph)
  for i in range(config.iteration):
#    batch_x, batch_y = dh.sample_pair(config.batch_size)
    loss_train_all = []
    loss_test_all = []
    permutated_indexes = np.random.permutation(x_train.shape[0])
    if i <= 200:
        learning_rate = 2e-4
    if i >200 and i<400:
        learning_rate = 1.5e-4
    if i >400:
        learning_rate = 1e-4
    for index in range(int(x_train.shape[0] / batch_size)):
        batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
        batch_x = x_train[batch_indexes]
        batch_y = y_train[batch_indexes]
    # =============================================================================
        dis_losses = []
        for _ in range(5):
            loss, _ = restore.dis_fit(batch_x, batch_y, learning_rate,learning_rate)
            dis_losses.append(loss)
#            print('gp is ', gp)
        print("iter dis loss{:>6d} : {}".format(i+1, np.mean(dis_losses)))
        loss = restore.gen_fit(batch_x, batch_y, learning_rate,learning_rate, index)
        loss_train_all.append(loss)
        print("iter gen loss{:>6d} : {}".format(i+1, np.mean(loss)))
    value = np.mean(loss_train_all)
    summary = tf.Summary(value=[tf.Summary.Value(tag="summary_tag", simple_value=value), ])
    train_writer.add_summary(summary,i) 
    print("iter {:>6d} : {}".format(i+1, np.mean(loss_train_all)))
#    reg.evaluate(batch_x, batch_y, i)    
    if (i+1) % 3 == 0:
        for index in range(int(x_test.shape[0] / batch_size)):
            batch_x = x_test[index*batch_size:(index+1)*batch_size]
            batch_y = y_test[index*batch_size:(index+1)*batch_size]
            loss = restore.evaluate(batch_x, batch_y)
            loss_test_all.append(loss)
        value = np.mean(loss_test_all)
        summary = tf.Summary(value=[tf.Summary.Value(tag="summary_tag",simple_value=value), ])
        test_writer.add_summary(summary,i)
        print("test iter {:>6d} : {}".format(i+1, value))
#      reg.deploy(config.tmp_dir, batch_x, batch_y)
    if (i+1) % 50 == 0:
        print('saving the result...')
        restore.save(config.ckpt_dir)
