# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:01:22 2018

@author: zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:40:10 2018

@author: zhang
"""

# pretrain the discriminator 
import os
import datetime
import click
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import load_images
from losses_freeway import adversarial_loss, generator_loss, wasserstein_loss, wasserstein_loss_new, perceptual_loss, perceptual_loss_100
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
BATCH_SIZE = 1
def save_all_weights(d, g, epoch_number):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, 'pretrain_{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_pretrain_{}.h5'.format(epoch_number)), True)
#    d.save_weights(os.path.join(save_dir, 'discriminator_pretrain_{}.h5'.format(epoch_number)), True)
def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
    data = load_images('..\\..\\dataset\\image_deform\\',n_images,istrain = True)
    y_train, x_train = data['B'], data['A']
# =============================================================================
    x_train = x_train[:,:,:,np.newaxis]
    y_train = y_train[:,:,:,np.newaxis]
#     
# =============================================================================
    
    tf_g = generator_model()
    tf_d = discriminator_model()
#    tf_g.load_weights('generator.h5')
#    d = discriminator_model()
#    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.ones((batch_size, 1))*0

    tf_g.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),loss = 'mean_squared_error')    
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
            
            g_loss = tf_g.train_on_batch([image_blur_batch],
                                               [image_full_batch])      
            print('batch {} dis_loss : {}'.format(index+1, np.mean(g_loss)))
            
        if epoch % 20 == 0:
            save_all_weights(tf_d, tf_g, epoch)
        #visulazation 
        if epoch % 10 == 0:
            generated_image = tf_g.predict(x=image_blur_batch, batch_size=1)
            generated_image = generated_image[0,:,:,0]
            generated_image = deprocess_image(generated_image)      
            plt.imshow(generated_image,cmap='gray')
            plt.show()
#    plt.plot(d_loss_out[40:])
#    plt.show()
#    plt.plot(g_loss_out)
#    plt.show()
@click.command()
@click.option('--n_images', default=500, help='Number of images to load for training')
@click.option('--batch_size', default=5, help='Size of batch')
@click.option('--epoch_num', default=100, help='Number of epochs for training')
@click.option('--critic_updates', default=10, help='Number of discriminator training')
def train_command(n_images, batch_size, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()








