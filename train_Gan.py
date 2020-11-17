# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:32:14 2018

@author: zhang
"""
permutated_indexes = np.random.permutation(x_train.shape[0])

import os
import datetime
import click
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import load_images
from losses_freeway import  wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss, perceptual_loss
from model_Gan import generator_model, discriminator_model, generator_containing_discriminator, generator_containing_discriminator_multiple_outputs

from keras.optimizers import Adam
import keras.backend as K
from functools import partial
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Add
from keras.layers.merge import _Merge
import matplotlib.pyplot as plt
from utils import load_images, deprocess_image
import time
from keras.models import load_model

BASE_DIR = 'weights/'
BATCH_SIZE = 5
def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


#def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
#    data = load_images('..\\..\\dataset\\image_deform\\',n_images,istrain = True)
#    y_train, x_train = data['B'], data['A']
## =============================================================================
##     im = Image.open('images//a1.png')
##     x_train = np.asarray(im)
#    x_train = x_train[:,:,:,np.newaxis]
#    y_train = y_train[:,:,:,np.newaxis]
##     im = Image.open('images//a2.png')
##     y_train = np.asarray(im)
##     
## =============================================================================
#    tf_g = generator_model()
#    tf_d = discriminator_model()
## restore the model    
##    tf_g.load_weights('generator.h5')
##    tf_d.load_weights('discriminator_pretrain.h5')
#    
#    d_on_g = generator_containing_discriminator_multiple_outputs(tf_g, tf_d)    
#    image_shape = (256, 256, 1)
#
#    img_blur  = Input(shape = image_shape)
#    img_clear = Input(shape = image_shape)
#    img_clear_gen = tf_g(img_blur) 
#    dis_img_clear = tf_d(img_clear)
#    dis_img_clear_gen = tf_d(img_clear_gen)
#
#    averaged_samples = RandomWeightedAverage()([img_clear, img_clear_gen])
#    averaged_dis =  tf_d(averaged_samples)
#    
#    partial_gp_loss = partial(gradient_penalty_loss,
#                              averaged_samples=averaged_samples,
#                              gradient_penalty_weight=10)
#    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
#     
#    discriminator_model2 = Model(inputs=[img_clear, img_blur],
#                                outputs=[dis_img_clear,dis_img_clear_gen,averaged_dis ])
#    tf_d.trainable = True  
#    discriminator_model2.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
#                            loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss])
#    tf_d.trainable = False    
#           
#    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    loss = [perceptual_loss, wasserstein_loss]
#    loss_weights = [100, 1]
#    d_on_g.compile(optimizer = d_on_g_opt, loss = loss, loss_weights = loss_weights)
#    tf_d.trainable = True  
#
#                                  
##    tf_d.trainable = False  
#    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.ones((batch_size, 1))*-1
#    penalty_zero_batch = np.zeros((batch_size, 1))
#    
#    d_loss_out = []
#    g_loss_out = []
#    for epoch in range(epoch_num):
#        print('epoch: {}/{}'.format(epoch, epoch_num))
#        print('batches: {}'.format(x_train.shape[0] / batch_size))
#
#        permutated_indexes = np.random.permutation(x_train.shape[0])
#        d_on_g_losses = []
#        d_losses = []
#        for index in range(int(x_train.shape[0] / batch_size)):
#            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
#            image_blur_batch = x_train[batch_indexes]
#            image_full_batch = y_train[batch_indexes]
#            
#            for _ in range(critic_updates):
#                loss_value = discriminator_model2.train_on_batch([image_full_batch, image_blur_batch],
#                                                   [output_true_batch, output_false_batch, penalty_zero_batch])
#                d_losses.append(loss_value)
#            print('batch {} dis_loss : {}'.format(index+1, np.mean(d_losses)))
#            d_loss_out.append(np.mean(d_losses))
#            tf_d.trainable = False
#
#            start_time = time.time()
#            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
#            time_elapsed = time.time() - start_time
#            d_on_g_losses.append(d_on_g_loss)
#            g_loss_out.append(d_on_g_loss[0])
#            print('batch {} g_out_loss : {}, time is: {}'.format(index+1, d_on_g_loss,time_elapsed))
#            tf_d.trainable = True
#            
##        with open('log.txt', 'a') as f:
##            f.write('{} - {} - {}\n'.format(epoch, loss_value, d_on_g_losses))        
#        save_all_weights(tf_d, tf_g, epoch, int(np.mean(d_on_g_losses)))
#        #visulazation 
#        if epoch % 10 == 0:
#            generated_image = tf_g.predict(x=image_blur_batch, batch_size=1)
#            generated_image = generated_image[0,:,:,0]
#            generated_image = deprocess_image(generated_image)      
#            plt.imshow(generated_image,cmap='gray')
#            plt.show()
#    plt.plot(d_loss_out[40:])
#    plt.show()
#    plt.plot(g_loss_out)
#    plt.show()
#@click.command()
#@click.option('--n_images', default=20, help='Number of images to load for training')
#@click.option('--batch_size', default=5, help='Size of batch')
#@click.option('--epoch_num', default=100, help='Number of epochs for training')
#@click.option('--critic_updates', default=10, help='Number of discriminator training')
#def train_command(n_images, batch_size, epoch_num, critic_updates):
#    return train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates)


if __name__ == '__main__':
#    train_command()
    n_images = 20
    batch_size = 5
    epoch_num = 50
    critic_updates = 10
    data = load_images('..\\..\\dataset\\image_deform\\',n_images,istrain = True)
    y_train, x_train = data['B'], data['A']
# =============================================================================
#     im = Image.open('images//a1.png')
#     x_train = np.asarray(im)
    x_train = x_train[:,:,:,np.newaxis]
    y_train = y_train[:,:,:,np.newaxis]
#     im = Image.open('images//a2.png')
#     y_train = np.asarray(im)
#     
# =============================================================================
    image_shape = (256, 256, 1)
    img_blur  = Input(shape = image_shape)
    tf_g = generator_model()
    tf_d = discriminator_model()
# restore the model    
#    tf_g.load_weights('generator.h5')
#    tf_d.load_weights('discriminator_pretrain.h5')
    
    d_on_g = generator_containing_discriminator_multiple_outputs(tf_g, tf_d)    

    img_blur  = Input(shape = image_shape)
    img_clear = Input(shape = image_shape)
    img_clear_gen = tf_g(img_blur) 
    dis_img_clear = tf_d(img_clear)
    dis_img_clear_gen = tf_d(img_clear_gen)

    averaged_samples = RandomWeightedAverage()([img_clear, img_clear_gen])
    averaged_dis =  tf_d(averaged_samples)
    
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=10)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
     
    discriminator_model2 = Model(inputs=[img_clear, img_blur],
                                outputs=[dis_img_clear,dis_img_clear_gen,averaged_dis ])
    tf_d.trainable = True  
    discriminator_model2.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,wasserstein_loss,partial_gp_loss])
    tf_d.trainable = False    
           
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer = d_on_g_opt, loss = loss, loss_weights = loss_weights)
    tf_d.trainable = True  
    
    # just for test 
    generator_test = Model(inputs = img_blur,outputs=img_clear_gen)
    generator_test.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),loss = 'mean_squared_error')    

                                  
#    tf_d.trainable = False  
    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.ones((batch_size, 1))*-1
    penalty_zero_batch = np.zeros((batch_size, 1))
    
    d_loss_out = []
    g_loss_out = []
    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch, epoch_num))
        print('batches: {}'.format(x_train.shape[0] / batch_size))

        permutated_indexes = np.random.permutation(x_train.shape[0])
        d_on_g_losses = []
        d_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]
            
#            for _ in range(critic_updates):
#                loss_value = discriminator_model2.train_on_batch([image_full_batch, image_blur_batch],
#                                                   [output_true_batch, output_false_batch, penalty_zero_batch])
#                d_losses.append(loss_value)
#            print('batch {} dis_loss : {}'.format(index+1, np.mean(d_losses)))
#            d_loss_out.append(np.mean(d_losses))
            tf_d.trainable = False

#            start_time = time.time()
#            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
#            time_elapsed = time.time() - start_time
#            d_on_g_losses.append(d_on_g_loss)
#            g_loss_out.append(d_on_g_loss[0])
#            print('batch {} g_out_loss : {}, time is: {}'.format(index+1, d_on_g_loss,time_elapsed))
            g_loss = generator_test.train_on_batch([image_blur_batch],
                                   [image_full_batch])      

            tf_d.trainable = True
            
#        with open('log.txt', 'a') as f:
#            f.write('{} - {} - {}\n'.format(epoch, loss_value, d_on_g_losses))        
        save_all_weights(tf_d, tf_g, epoch, int(np.mean(d_on_g_losses)))
        #visulazation 
        if epoch % 10 == 0:
            generated_image = tf_g.predict(x=image_blur_batch, batch_size=1)
            generated_image = generated_image[0,:,:,0]
            generated_image = deprocess_image(generated_image)      
            plt.imshow(generated_image,cmap='gray')
            plt.show()
    plt.plot(d_loss_out[40:])
    plt.show()
    plt.plot(g_loss_out)
    plt.show()

