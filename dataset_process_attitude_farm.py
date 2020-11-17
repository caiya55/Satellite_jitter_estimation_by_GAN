# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:24:57 2018

@author: zhang
"""

import numpy as np
import os
from PIL import Image
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from numpy.random import random_sample
from numpy import sin
from random import random
# data：2018/5/27

def obtain_classes():
    path = 'images/'
    sub = True
    classes = []
    for root, dirs, files in os.walk(path):
        if sub:
            for cla in dirs:
                classes.append(cla)
        sub = False        
    return classes


def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False

def list_image_files(directory):
    files = os.listdir(directory)
    img_list = [os.path.join(directory, f) for f in files if is_an_image_file(f)]
    name = [f for f in files]
    return img_list, name
def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((256,256))
    return img
def preprocess(img):
    img = np.array(img) 
    img = img / 255
    return img
def jitter(width):
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.05, 0.02, 0.01];    
    rf = random()*0.1 + 0.15
    f = [f*rf for f in f]
#    pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
        
    amp = [1, 0.1, 0.05, 0.01]
    ra = random()*2 + 2
    amp = [amp*ra for amp in amp]
#    fy = 0.2
#    b = 1
    jix  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
#    jix = sin(f[0]*x)
#    jix = jix - jix.mean()
#    jix[0:5] = 0
#    jix[-5:] = 0
    return jix



def jitter_poly(width):
    x = np.arange(0,1, 1/width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    rf = random()*0.3 + 0.2
    f = [f*rf for f in f]
    pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
        
    amp = np.ones(4)
    mu = 2
    amp[0] = (random()*(-2) + 1)*mu
    amp[1] = (random()*(-2) + 1)*mu
    amp[2] = (random()*(-2) + 1)*mu
    amp[3] = (random()*(-2) + 1)*mu


#    fy = 0.2
#    b = 1
    jix  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
    jix = amp[3]*(x)**3 + amp[2]*(x)**2 + amp[1]*(x) + amp[0] 
#    jix = sin(f[0]*x)
    jix = jix
#    jix[0:5] = 0
#    jix[-5:] = 0
    return jix

def jitter_y(width):
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    rf = random()*0.1 
    f = [f*rf for f in f]
    pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
    
    amp = [1, 0.3, 0.1, 0.05]
    ra = random()*0.1 + 0.1 
    amp = [amp*ra for amp in amp]
#    fy = 0.2
#    b = 1
    jix  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
#    jix = sin(f[0]*x)
    jix = jix
#    jix[0:5] = 0
#    jix[-5:] = 0
    return jix

def jitter_sine(width, fe, att):
    x = np.arange(width)
    jix = 2*att*sin(fe*x)
    jix = jix
    jix[0:5] = 0
    jix[-5:] = 0
    return jix

def jitter_maker_4sin(imgs):
    imgs = imgs[:,:,:,0]
    width = imgs.shape[1]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    imgs_out = np.zeros_like(imgs)
    jix_out = np.zeros([imgs.shape[0], imgs.shape[1]])
    for i in range(imgs.shape[0]):
#        jix = jitter(width)  
        jix = jitter_poly(width)
        f = interp2d(x, y, imgs[0], kind='linear')
        jix_out[i] = jix
        for index in range(width):
            out_tmp = f(y+jix[index], x[index]).T
            imgs_out[i, index] = out_tmp
    imgs_out = imgs_out[:,:,:,np.newaxis]       
    jix_out = jix_out[:,:,np.newaxis]
    return imgs_out, jix_out

def jitter2D(imgs):
    imgs = imgs[:,:,:,0]
    width = imgs.shape[1]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    imgs_out = np.zeros_like(imgs)
    jix_out_x = np.zeros([imgs.shape[0], imgs.shape[1]])
    jix_out_y = np.zeros([imgs.shape[0], imgs.shape[1]])
    for i in range(imgs.shape[0]):
        jix = jitter(width) 
        jiy = jitter_y(width) 
        f = interp2d(x, y, imgs[i], kind='linear')
        jix_out_x[i] = -jix # reverse
        jix_out_y[i] = -jiy
        for index in range(width):
            out_tmp = f(y+jix[index], x[index]+jiy[index]).T
            imgs_out[i, index] = out_tmp
    imgs_out = imgs_out[:,:,:,np.newaxis]       
    jix_out_x = jix_out_x[:,:,np.newaxis]
    jix_out_y = jix_out_y[:,:,np.newaxis]
    jix_out = np.concatenate([jix_out_x,jix_out_y],axis=-1)
    return imgs_out, jix_out

def jitter_with_curve(imgs, curves):
    imgs = imgs[:,:,:,0]
    width = imgs.shape[1]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    imgs_out = np.zeros_like(imgs)
    jix_out_x = np.zeros([imgs.shape[0], imgs.shape[1]])
    jix_out_y = np.zeros([imgs.shape[0], imgs.shape[1]])
    for i in range(imgs.shape[0]):
        jix = curves[i,:,0] 
        jiy = curves[i,:,1] 
        f = interp2d(x, y, imgs[i], kind='linear')
        jix_out_x[i] = jix
        jix_out_y[i] = jiy
        for index in range(width):
            out_tmp = f(y+jix[index], x[index]+jiy[index]).T
            imgs_out[i, index] = out_tmp
    imgs_out = imgs_out[:,:,:,np.newaxis]       
    jix_out_x = jix_out_x[:,:,np.newaxis]
    jix_out_y = jix_out_y[:,:,np.newaxis]
    jix_out = np.concatenate([jix_out_x,jix_out_y],axis=-1)
    return imgs_out, jix_out

def jitter_maker(img_gray, fe, att):
#    img = preprocess(img_gray)
    img = img_gray
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    f = interp2d(x, y, img, kind='linear')
    
    img_out = np.zeros([fe.shape[0], img.shape[0], img.shape[1]])
    jix_out = np.zeros([fe.shape[0], img.shape[0]])
    for i in range(fe.shape[0]):
        jix = jitter_sine(width, fe[i], att)  
        jix_out[i] = jix
        for index in range(img.shape[0]):
            out_tmp = f(y+jix[index], x[index]).T
            img_out[i,index] = out_tmp
#    img_out = img_out * 255.
#    img_out = img_out.astype(np.uint8)
    
    return img_out, jix_out
# main loop 
#classes = obtain_classes()  
#for cla in classes:
if __name__ == "__main__":

    cla = 'freeway'
    rootDir = 'images/' + cla + '/'
    save_dir_jit = 'image_deform/' + cla + '/jitter/'
    
    save_dir_A_test = 'image_deform/' + cla + '/A_test_attitude/'
    save_dir_B_test = 'image_deform/' + cla + '/B_test_attitude/'
    
    if not os.path.exists(save_dir_A_test):
        os.makedirs(save_dir_A_test)
        os.makedirs(save_dir_B_test)
        os.makedirs(save_dir_jit)
    list_dirs = os.walk(rootDir) 
    image_list,name = list_image_files(rootDir)
    
    
    for i in range(0,20):  # just random
        print(i)
        img_gray = load_image(image_list[i])
        img = preprocess(img_gray)
        width = img.shape[0]
        x = np.arange(width).astype(float)
        y = np.arange(width).astype(float)
        
        f = interp2d(x, y, img, kind='linear')
        jix = jitter()
        x_i = x 
        y_i = y + jix
        
        # make a loop
        img_out = np.zeros(img.shape)
        for index in range(img.shape[0]):
            out_tmp = f(y+jix[index], x[index]).T
            img_out[index] = out_tmp
        img_out = img_out * 255.
        img_out = img_out.astype(np.uint8)
        
        # save the image
        save_path_A = os.path.join(save_dir_A_test, name[i])
        save_path_B = os.path.join(save_dir_B_test, name[i])
        jitter_path = os.path.join(save_dir_jit, name[i])+'.npy'
    
        Im = Image.fromarray(img_out)
        np.save(jitter_path,jix)
        Im.save(save_path_A)  
        img_gray.save(save_path_B)  

## plot the image    
#arr = img_out * 255.
#arr = arr.astype(np.uint8)
#plt.imshow(arr,cmap='gray')
#plt.show()
#out1 = f(x_i, y)
#arr = out1 * 255.
#arr = arr.astype(np.uint8)
#plt.imshow(arr,cmap='gray')
#plt.show()
#plt.plot(x,jix_1)
#plt.show()