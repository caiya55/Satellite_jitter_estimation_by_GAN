# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:43:09 2018

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
from dataset_process_attitude import jitter_maker_4sin, jitter2D, jitter_with_curve
from PIL import Image
from random import randint
import cv2
import argparse

n_images = 200

def write_log(callback, names, logs, batch_no):  
    for name, value in zip(names, logs):  
        summary = tf.Summary()  
        summary_value = summary.value.add()  
        summary_value.simple_value = value  
        summary_value.tag = name  
        callback.writer.add_summary(summary, batch_no)  
        callback.writer.flush()  
        
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
parser.add_argument("--kernel_size", type=int, help="choose size of the kernel", default = 3)

args = parser.parse_args()

if __name__ == "__main__":
#  main()
  sess = tf.Session()   
  config = get_config(is_train=True)
#  tf.get_variable_scope().reuse_variables()
  config.ckpt_dir = 'gan_ckpt'

  mkdir(config.tmp_dir) 
  mkdir(config.ckpt_dir)

  restore = D_on_G(sess, config, "DIRNet", args, is_train=True)
  restore.restore(config.ckpt_dir)  
#  dh = MNISTDataHandler("MNIST_data", is_train=True)
  data = load__class_images_with_C('..\\..\\dataset\\image_deform\\freeway\\',30,istrain = True) # load the images from different classes
#  data = load_images('..\\..\\dataset\\image_deform\\',600,istrain = True) # load the images from different classes

  y_train, x_train, z_train = data['B'], data['A'], data['C']
  x_train = x_train[:,:,:,np.newaxis]
  y_train = y_train[:,:,:,np.newaxis] 
#  z_train = z_train[:,:,:,np.newaxis]
  # load the test dataset
  data = load__class_images_with_C('..\\..\\dataset\\image_deform\\freeway\\',10,istrain = False) # load the images from different classes
#  data = load_images('..\\..\\dataset\\image_deform\\',100,istrain = False)# load the images from different classes
  y_test, x_test, z_test = data['B'], data['A'], data['C']
  x_test = x_test[:,:,:,np.newaxis]
  y_test = y_test[:,:,:,np.newaxis]
#  z_test = z_test[:,:,:,np.newaxis]
  
  train_writer = tf.summary.FileWriter('gan/train', sess.graph)
  test_writer = tf.summary.FileWriter('gan/test')
#  all_writer = tf.summary.FileWriter('cnn/all')
#  merged_summary = tf.summary.merge_all()  
#  all_writer.add_graph(sess.graph)

  batch_size = config.batch_size
  kernel_past = []
  validation_loss = []
#  batch_x = x_train
#  batch_y = y_train
#  batch_x_out = jitter_with_curve(batch_y, z_train)
  bo = 0
  for i in range(config.iteration):
#    batch_x, batch_y = dh.sample_pair(config.batch_size)
    loss_train_all = []
    loss_test_all = []
    permutated_indexes = np.random.permutation(x_train.shape[0])
    if i <= 20:
        learning_rate = 1.3e-4
    if i >20 and i<40:
        learning_rate = 0.6e-4
    if i > 40:
        learning_rate = 0.3e-4
#    if i >400 and i%200==0 :
#        learning_rate = learning_rate-0.4e-4
    for index in range(int(x_train.shape[0] / batch_size)):
        batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
        batch_y = y_train[batch_indexes] 
        batch_x = x_train[batch_indexes]
        if i < 20:
            batch_z = z_train[batch_indexes]
            batch_z = -batch_z[:,bo:bo+256,:]
        else:
            batch_x, batch_z = jitter2D(batch_y)   
        batch_x = batch_x[:,bo:bo+256,bo:bo+256,:]
        batch_y = batch_y[:,bo:bo+256,bo:bo+256,:]
    # =============================================================================
        dis_losses = []
        
        
#        for _ in range(5):
#            loss, _ = restore.dis_fit(batch_x, batch_y, learning_rate,learning_rate)
#            dis_losses.append(loss)
##            print('gp is ', gp)
#        print("iter dis loss{:>6d} : {}".format(i+1, np.mean(dis_losses)))
        loss, wrap = restore.gen_fit(batch_x, batch_y, batch_z, learning_rate,learning_rate, i)
        loss_train_all.append(loss)
#        print("iter gen loss{:>6d} : {}".format(i+1, np.mean(loss)))
#    if i % 20 == 0:
#        print(kernel[0,:,:,0])
    value = np.mean(loss_train_all)
    summary = tf.Summary(value=[tf.Summary.Value(tag="summary_tag", simple_value=value), ])
    train_writer.add_summary(summary,i) 
    print("iter {:>6d} : {}".format(i+1, value))
#    reg.evaluate(batch_x, batch_y, i)    
    if (i+1) % 1 == 0:
        for index in range(int(x_test.shape[0] / batch_size)):
            batch_x = x_test[index*batch_size:(index+1)*batch_size,bo:bo+256,bo:bo+256,:]
            batch_y = y_test[index*batch_size:(index+1)*batch_size,bo:bo+256,bo:bo+256,:]
            batch_z = -z_test[index*batch_size:(index+1)*batch_size,bo:bo+256,:]
            loss = restore.evaluate(batch_x, batch_y, batch_z)     
            loss_test_all.append(loss)
        value = np.mean(loss_test_all)
        summary = tf.Summary(value=[tf.Summary.Value(tag="summary_tag",simple_value=value), ])
        test_writer.add_summary(summary,i)
        print("test iter {:>6d} : {}".format(i+1, value))
        validation_loss.append(value)
##      reg.deploy(config.tmp_dir, batch_x, batch_y)
    if (i+1) % 20 == 0:
        print('saving the result...')
        restore.save(config.ckpt_dir)
    means = []
    if (i+1) % 2 == 0:
        img_i = 0
        _,_, y_result, wrap_test = restore.predict_one(batch_x[img_i], config)
        plt.plot(batch_z[img_i,:,0])
        plt.plot(wrap_test[:,0])
        plt.title('Real Curve vs Obtained Curve')
        plt.show()

#        plt.imshow(y_result[:,:,0], cmap='gray')
#        plt.grid(False)
#        plt.axis('off')
#        plt.title('Restored image')
#        plt.show()            
#
#        plt.imshow(batch_x[img_i,:,:,0],cmap='gray')
#        plt.grid(False)
#        plt.axis('off')
#        plt.title('Raw image')
#        
#        plt.imshow(batch_y[img_i,:,:,0],cmap='gray')
#        plt.grid(False)
#        plt.axis('off')
#        plt.title('GroundTruth')
#        plt.show()
        
        yaogan_x = preprocess('..//..//dataset//yaogan26//for_classification_air.png', 0, 0)
        loss1, loss2, output,wrap_yaogan = restore.predict_one(yaogan_x[0], config)        
              
        plt.imshow(output[:,:,0], cmap='gray')
        plt.grid(False)
        plt.axis('off')   
        plt.title('Rstored image')
#        plt.savefig('results//restored1.pdf')
        cv2.imwrite('results//restored3.png', uint_img(output[:,:,0]))

        plt.show()            

        plt.imshow(yaogan_x[0,:,:,0], cmap='gray')
        plt.grid(False)
        plt.axis('off')   
        plt.title('Raw image')
        cv2.imwrite('results//raw3.png', uint_img(yaogan_x[0,:,:,0]))
        plt.show()            
    loss_name =  str(args.max_pooling)+'_'+str(args.final_layer)+'_'+str(args.alpha)+'_'+str(args.kernel_size)+'.npy'           
    np.save(loss_name, validation_loss)


#        plt.plot(wrap_yaogan[:,0])
#        plt.title('Obtained attitude cross-track')
#        plt.grid(True)
#        plt.xlabel('Lines')
#        plt.ylabel('Pixel')
#        plt.savefig('results//cross-track-curve3.png')
#        plt.show()
#
#        plt.plot(wrap_yaogan[:,1])
#        plt.title('Obtained attitude along-track')
#        plt.grid(True)
#        plt.xlabel('Lines')
#        plt.ylabel('Pixel')
#        plt.savefig('results//along-track-curve3.png')
#        plt.show()

#        plt.plot(batch_z[img_i,:,0])
#        plt.plot(wrap_test[:,0])
#        plt.title('Real Curve')
#        plt.show()
