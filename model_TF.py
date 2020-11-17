# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:10:28 2018

@author: zhang
"""

import tensorflow as tf
from WarpST_one import WarpST_one 
from ops import *
#from bicubic_interp import bicubic_interp_2d
#from network import restore_net
import numpy as np
from keras.layers import Input, concatenate, Activation, Add, UpSampling2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Convolution2D, Conv2DTranspose
from keras.layers.core import Dropout, Dense, Flatten, Lambda
from keras.layers.merge import Average
from keras.layers import BatchNormalization
from keras.models import Model, Sequential
import keras.backend as K
from layer_utils import ReflectionPadding2D, res_block
from subpixel import SubpixelConv2D
import random
from losses_freeway import  wasserstein_loss_TF, gradient, perceptual_loss,gradient2, l1_loss, l2_loss
from functools import partial
from bicubic_interp import bicubic_interp_2d


ngf = 32
ndf = 64
input_nc = 1
output_nc = 1
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
n_blocks_gen = 4
imsize=[256,256]
def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
  return tf.contrib.layers.batch_norm(x, 
    decay=momentum,
    updates_collections=None,
    epsilon=epsilon,
    scale=True,
    is_training=is_train, 
    scope=name)
  
def conv_layer(inputs,
               channels_in,    # 输入通道数
               channels_out,   # 输出通道数
               kernel_size,   # kernel size
               padding='SAME',  
               name='conv'):   # 名称
    with tf.name_scope(name):
        kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, channels_in, channels_out],
                                                  stddev=0.05), name='W')
        biases = tf.Variable(tf.constant(0.05, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(inputs, filter=kernel, strides=[1, 1, 1, 1], padding=padding)
        act = tf.nn.relu(conv + biases)

        # 收集以下三个信息，统计直方图
        tf.summary.histogram('weights', kernel)   
        tf.summary.histogram('biases', biases)     
        tf.summary.histogram('activations', act)
    with tf.variable_scope('visualization'):
        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)    
        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])   
        k1 = tf.transpose(kernel, [3, 0, 1, 2])
        # this will display random 3 filters from the 64 in conv1
        
        tf.summary.image('conv1/filters', kernel_transposed[...,13:14], max_outputs=16)
#        layer1_image1 = act[0:1, :, :, 0:16]
#        layer1_image1 = tf.transpose(layer1_image1, perm=[3,1,2,0])
#        tf.summary.image("filtered_images_layer1", layer1_image1, max_outputs=16)
        kernel_show = k1[...,13:14]
        return act, k1
    
#def generator_model_2(inputs, istrain, reuse):
#    """Build generator architecture."""
#    # inputs: tensor with shape [bn, 256,256, 1]
##    inputs = Input(shape=input_shape_generator)
#    with tf.variable_scope('gen_', reuse=reuse):
#        x = ReflectionPadding2D((3, 3))(inputs)
#        x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
#        x = batch_norm(x, "bn1", is_train=istrain)
#        x = Activation('relu')(x)
##        x = conv_layer(x, 1, 16, )
#            
#        n_downsampling = 3
#        for i in range(n_downsampling):
#            mult = 2**i
#            x = Conv2D(filters=ngf*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
##            x = BatchNormalization()(x, training=istrain)         
#            x = batch_norm(x, "down_bn_"+str(i), is_train=istrain)
#            tf.summary.histogram('before_active', x)
#            x = Activation('relu')(x)
#            tf.summary.histogram('after_activate', x)
#        mult = 2**n_downsampling
#        for i in range(n_blocks_gen):
#            x = res_block(x, ngf*mult, use_dropout=True)
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            x = UpSampling2D()(x)
#            x = Conv2D(filters=int(ngf * mult / 2),kernel_size=(3,3),padding='same')(x)        
##            x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
##            x = BatchNormalization()(x, training=istrain)
#            x = batch_norm(x, "up_bn_"+str(i), is_train=istrain)
#            x = LeakyReLU(alpha=0.3)(x)
#
#        x = Conv2D(filters=2, kernel_size=(9,9), padding='same')(x)
#        x = batch_norm(x, "final", is_train=istrain)
#        wrap = Activation('sigmoid')(x)
#        wrap = tf.multiply(tf.add(wrap,-0.5), 16)
##        x, _ = conv_layer(x, 32, 2, kernel_size=9, padding='SAME')
##        x_mean = tf.reduce_mean(x, axis=2)
##        x = tf.expand_dims(x_mean, 2)
##        wrap = tf.tile(x, multiples=[1, 1, 256, 1])
#        outputs = Lambda(WarpST_one, arguments={'inputs':inputs, 'name':str(random.random())})(wrap)
#    #    outputs = Add()([x, inputs])
#    
#    #    model = Model(inputs=inputs, outputs=outputs, name='Generator')
#        # tf only output the model
#        return outputs, wrap
ngf = 32
n_blocks_gen = 8
def generator_model(args, inputs, istrain, reuse):
    """Build generator architecture."""
    with tf.variable_scope('gen_', reuse=reuse):
        x = ReflectionPadding2D((3, 3))(inputs)
        x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
        x = batch_norm(x, "bn1", is_train=istrain)
        x = Activation('relu')(x)

#        x = MaxPooling2D((2, 2), padding='same')(x)   
#        x = Conv2D(filters=ngf, kernel_size=(7,7), padding='same')(x)
#        x = batch_norm(x, "bn2", is_train=istrain)
#        x = Activation('relu')(x)

#        x = ReflectionPadding2D((3, 3))(inputs)
#        x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
#        x = batch_norm(x, "bn1", is_train=istrain)
#        x = Activation('relu')(x)
        
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
#            x = ReflectionPadding2D((2, 2))(x)
            if args.max_pooling == False:
                x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='valid')(x)
            else:
                x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=1, padding='valid')(x)
                x = MaxPooling2D((2, 2), padding='valid')(x)
#            x = BatchNormalization()(x, training=istrain)         
            x = batch_norm(x, "down_bn_"+str(i), is_train=istrain)
            tf.summary.histogram('before_active', x)
            x = Activation('relu')(x)
            tf.summary.histogram('after_activate', x)
        mult = 2**n_downsampling
        for i in range(n_blocks_gen):
            x = res_block(x, ngf*mult, use_dropout=True)
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            x = UpSampling2D()(x)
#            x = Conv2D(filters=int(ngf * mult / 2),kernel_size=(3,3),padding='same')(x)        
##            x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
##            x = BatchNormalization()(x, training=istrain)
#            x = batch_norm(x, "up_bn_"+str(i), is_train=istrain)
#            x = LeakyReLU(alpha=0.3)(x)

        
        x = Conv2D(filters=2, kernel_size=(5, 5), padding='same')(x)        
        x = batch_norm(x, "final", is_train=istrain)
        wrap = Activation('sigmoid')(x)
        wrap = tf.multiply(tf.add(wrap,-0.5), 8)
        # dense layer
        dense = tf.layers.flatten(wrap)
        output_size = 128
        output_size = args.final_layer# we use the args value here to decide the final layer number
        dense_out = tf.layers.dense(inputs=dense, units=output_size*2)
        x_mean = tf.reshape(dense_out,[-1,output_size,2])
        x_mean = tf.expand_dims(x_mean, 2)
        wrap = tf.tile(x_mean, multiples=[1, 1, output_size, 1])
        wrap = bicubic_interp_2d(wrap, imsize)
        outputs = Lambda(WarpST_one, arguments={'inputs':inputs, 'name':str(random.random())})(wrap)
        return outputs, wrap[:,:,0,:]

    
def discriminator_model(inputs, istrain=False, reuse=True):
    """Build discriminator architecture."""
        # inputs: tensor with shape [bn, 256,256, 1]
    with tf.variable_scope('dis_', reuse=reuse):
        n_layers, use_sigmoid = 3, False
    #    inputs = Input(shape=input_shape_discriminator, name = 'dis_input')
    
        x = Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
    
        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
#            x = BatchNormalization()(x,training=istrain)
            x = batch_norm(x, "bn1_"+str(n), is_train=istrain)
            x = LeakyReLU(0.2)(x)
    
        nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
#        x = BatchNormalization()(x, training=istrain)
        x = batch_norm(x, "bn2", is_train=istrain)
        x = LeakyReLU(0.2)(x)
    
        x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)
    
        x = Flatten()(x)
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)
    #    x = K.mean(x)
    #    model = Model(inputs=inputs, outputs=x, name='Discriminator')
        outputs=x
        return outputs

def generator_containing_discriminator_multiple_outputs(inputs):
    # inputs: tensor with shape [bn, 256,256, 1]
#    inputs = Input(shape=image_shape)
    generated_image,_ = generator_model(inputs)
    outputs = discriminator_model(generated_image)
#    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return generated_image, outputs


class D_on_G(object):
  def __init__(self, sess, config, name, args, is_train):
    self.sess = sess
    self.name = name
    self.is_train = tf.placeholder(tf.bool)
    im_shape = [config.batch_size, config.im_size[0], config.im_size[1], 1]
    curve_shape = [config.batch_size, config.im_size[0], 2] # two dimension
    self.img_blur = tf.placeholder(tf.float32, im_shape)
    self.img_clear = tf.placeholder(tf.float32, im_shape)
    self.img_clear = tf.placeholder(tf.float32, im_shape)
    self.gen_lr = tf.placeholder(tf.float32)
    self.dis_lr = tf.placeholder(tf.float32)
    self.real_curve = tf.placeholder(tf.float32, curve_shape)
    #============tensorflow setting    
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        self.img_clear_gen, self.obtain_curve = generator_model(args, self.img_blur, self.is_train, reuse=False) 
        self.dis_img_clear = discriminator_model(self.img_clear, self.is_train, reuse=False)
        self.dis_img_clear_gen = discriminator_model(self.img_clear_gen,self.is_train, reuse=True)
    self.t_vars = tf.trainable_variables()
    self.d_vars = [var for var in self.t_vars if 'dis_' in var.name]
    self.g_vars = [var for var in self.t_vars if 'gen_' in var.name]
    # calculate the loss function
    self.gen_loss = -tf.reduce_mean(self.dis_img_clear_gen)
    self.dis_loss = -wasserstein_loss_TF(self.dis_img_clear, self.dis_img_clear_gen)
    self.curve_loss = l2_loss(self.real_curve, self.obtain_curve)
    self.curve_smooth_loss = l2_loss(self.obtain_curve[:,:-1,:], self.obtain_curve[:,1:,:])
#    self.grad_penalty = gradient2(self.img_clear_gen, self.img_clear, discriminator_model)    
    alpha = tf.random_uniform(
        shape=[config.batch_size, 1,1,1],minval=0.,maxval=1.)
    differences = self.img_clear_gen - self.img_clear
    interpolates = self.img_clear + (alpha * differences)
    gradients = tf.gradients(discriminator_model(interpolates,reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    self.grad_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    self.content_loss = perceptual_loss(self.img_clear, self.img_clear_gen)
    self.discriminator_loss = self.dis_loss + 10*self.grad_penalty 
    self.generator_loss = args.alpha*self.content_loss + (1-args.alpha)*self.curve_loss #+ 5*self.curve_smooth_loss
    
    # save the model    
    self.saver = tf.train.Saver()

    self.merged_summary = tf.summary.merge_all()
    
    # optimize the loss
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        self.gen_train_op = tf.train.AdamOptimizer(
            learning_rate=self.gen_lr,beta1=0.5,beta2=0.9).minimize(self.generator_loss,var_list=self.g_vars)
        self.dis_train_op = tf.train.AdamOptimizer(
            learning_rate=self.dis_lr,beta1=0.5,beta2=0.9).minimize(self.discriminator_loss,var_list=self.d_vars)

#    tf.summary.scalar('loss', self.generator_loss)
    self.summary = tf.summary.merge_all()
    self.writer = tf.summary.FileWriter('cnn/all', sess.graph)
    self.sess.run(
      tf.global_variables_initializer())
    
  def gen_fit(self, batch_x, batch_y, batch_z, gen_lr, dis_lr, i):
      
    _, loss,summary, kernel  = \
      self.sess.run([self.gen_train_op, self.generator_loss, self.merged_summary, self.obtain_curve], 
      {self.img_blur:batch_x, self.img_clear:batch_y, self.real_curve:batch_z, self.is_train: True, self.gen_lr :gen_lr, self.dis_lr :dis_lr })
    self.writer.add_summary(summary, i)
    return loss, kernel

  def dis_fit(self, batch_x, batch_y, gen_lr, dis_lr):
    _, loss, gp = \
      self.sess.run([self.dis_train_op, self.discriminator_loss, self.grad_penalty], 
      {self.img_blur:batch_x, self.img_clear:batch_y, self.is_train: True, self.gen_lr :gen_lr, self.dis_lr :dis_lr })
    return loss, gp

  def evaluate(self, batch_x, batch_y, batch_z):
    loss = self.sess.run([self.generator_loss], 
                         {self.img_blur:batch_x, self.img_clear:batch_y,
                          self.real_curve:batch_z, self.is_train: False, self.gen_lr :1e-4,self.dis_lr :1e-4})
    return loss

  def predict(self, batch_x, batch_y, batch_z):
    loss, gen, kernel  = self.sess.run([self.generator_loss, self.img_clear_gen, self.obtain_curve],
                       {self.img_blur:batch_x, self.img_clear:batch_y,
                        self.real_curve:batch_z, self.is_train: False, self.gen_lr :1e-6, self.dis_lr :1e-6})
    return loss, gen, kernel
#      
  def predict_one(self, batch_x, config):

    batch_y = np.zeros([config.batch_size,256,256,1])
    batch_z = np.zeros([config.batch_size,256,2])
    batch_x_in = np.zeros([config.batch_size,256,256,1])
    batch_x_in[0] = batch_x
    loss1,loss2, gen, wrap  = self.sess.run([self.content_loss, self.curve_loss, self.img_clear_gen, self.obtain_curve],
                       {self.img_blur:batch_x_in, self.img_clear:batch_y, 
                        self.real_curve: batch_z, self.is_train: False, self.gen_lr :1e-6, self.dis_lr :1e-6})
    return loss1 ,loss2,  gen[0], wrap[0]
#
  def show(self, x, y):
    z  = self.sess.run(self.z, {self.x:x, self.y:y, self.is_train: False})
    return z
  def save(self, dir_path):
    self.saver.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.saver.restore(self.sess, dir_path+"/model.ckpt")
    
