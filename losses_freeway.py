# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:35:37 2018

@author: zhang
"""

import keras.backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import concatenate
import tensorflow as tf
from keras.layers.merge import _Merge

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
    y_true_3 = concatenate([y_true,y_true,y_true],axis=-1)
    y_pred_3 = concatenate([y_pred,y_pred,y_pred],axis=-1)
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    # let the loss model can't be trained
    loss_model.trainable = False
    # loss_model.summary()
#    return K.mean(K.square(loss_model(y_true_3) - loss_model(y_pred_3)))
    return K.mean(K.square((y_true) - (y_pred)))



def generator_loss(y_true, y_pred):
    return K_1 * perceptual_loss(y_true, y_pred) + K_2 * l1_loss(y_true, y_pred)

def adversarial_loss(y_true, y_pred):
    return -K.log(y_pred)

def wasserstein_loss(y_true, y_pred):
#    return K.mean(y_true*y_pred)
    return K.mean(y_true)*K.mean(y_pred)

def wasserstein_loss_TF(real, generate):
    return tf.reduce_mean(real) - tf.reduce_mean(generate)

def wasserstein_loss_new_g(real, generate):
    return K.mean(real) - K.mean(generate)


# gradient penalty
def gradient(fake,real, discriminator):
    alpha = tf.random_uniform((tf.shape(fake)[0], 1, 1, 1),
                              minval = 0., maxval = 1,)
    differ = fake - real
    interp = real + (alpha * differ)
    grads = tf.gradients(discriminator(interp, True, True), [interp])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                   reduction_indices = [3]))
    grad_penalty = tf.reduce_mean((slopes - 1.)**2)
    return grad_penalty    


def gradient2(fake_data,real_data, discriminator):
    alpha = tf.random_uniform(
            shape=[tf.shape(fake_data)[0], 1], minval=0.,maxval=1.)        
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty   
def average_weighted_image(fake_data, real_data,batch_size):
#    alpha = tf.random_uniform(
#            shape=[tf.shape(fake_data)[0], 1], minval=0.,maxval=1.)     
    alpha = K.random_uniform((batch_size, 1, 1, 1))
    return Add()([(alpha * fake_data) , ((1 - alpha) * real_data)])   
 
BASE_DIR = 'weights/'
BATCH_SIZE = 5

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

