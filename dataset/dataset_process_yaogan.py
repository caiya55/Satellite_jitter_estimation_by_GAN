# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:50:56 2018

@author: zhang
"""

import numpy as np
import os
from PIL import Image
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from numpy.random import random_sample
from numpy.random import randint
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
    # low down the resolution of the image
    return img
def preprocess(img):
    img = np.array(img) 
    img = img / 255
#    img = (img - 127.5) / 127.5
    return img
def jitter(width):
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    rf = random() + 0.8
    f = [f*rf for f in f]   #fre is 0.05-0.2
    #pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
    
    amp = [1, 0.2, 0.1, 0.05]
    ra = random()*1 + 0.5 
    amp = [amp*ra for amp in amp]
#    fy = 0.2
#    b = 1
    jix_1  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
    #jix = b*sin(fy*y+b)
    jix = jix_1
    jix[0:5] = 0
    jix[-5:] = 0
    return jix

def jitter_x(width):
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    rf = random() + 0.6
    f = [f*rf for f in f]   #fre is 0.05-0.2
    #pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
    
    amp = [1, 0.2, 0.1, 0.05]
    ra = random()*0.5 + 3.5 
    amp = [amp*ra for amp in amp]
#    fy = 0.2
#    b = 1
    jix_1  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
    #jix = b*sin(fy*y+b)
    jix = jix_1
    jix[0:5] = 0
    jix[-5:] = 0
    return jix

def image_show(img):
    arr = subimg_deform * 255.
    arr = arr.astype(np.uint8)
    plt.imshow(arr,cmap='gray')
    plt.grid(False)
    plt.axis('off')
    plt.show()
   
#def jitter2D(imgs):
#    imgs = imgs[:,:,:,0]
#    width = imgs.shape[1]
#    x = np.arange(width).astype(float)
#    y = np.arange(width).astype(float)
#    
#    imgs_out = np.zeros_like(imgs)
#    jix_out_x = np.zeros([imgs.shape[0], imgs.shape[1]])
#    jix_out_y = np.zeros([imgs.shape[0], imgs.shape[1]])
#    for i in range(imgs.shape[0]):
#        jix = jitter(width) 
#        jiy = jitter(width) 
#        f = interp2d(x, y, imgs[i], kind='linear')
#        jix_out_x[i] = jitter_x
#        jix_out_y[i] = jiy
#        for index in range(width):
#            out_tmp = f(y+jix[index], x[index]+jiy[index]).T
#            imgs_out[i, index] = out_tmp
#    imgs_out = imgs_out[:,:,:,np.newaxis]       
#    jix_out_x = jix_out_x[:,:,np.newaxis]
#    jix_out_y = jix_out_y[:,:,np.newaxis]
#    jix_out = np.concatenate([jix_out_x,jix_out_y],axis=-1)
#    return imgs_out, jix_out

def jitter2D_single(img):
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    img_out = np.zeros_like(img)
    jix = jitter_x(width) 
    jiy = jitter(width) 
    f = interp2d(x, y, img, kind='linear')
    for index in range(width):
        out_tmp = f(y+jix[index], x[index]+jiy[index]).T
        img_out[index] = out_tmp
    jix_out_x = jix[:,np.newaxis]
    jix_out_y = jiy[:,np.newaxis]
    jix_out = np.concatenate([jix_out_x,jix_out_y],axis=-1)
    return img_out, jix_out

# main loop 
    
def run():
    #classes = obtain_classes()  
    #for cla in classes:
    cla = 'yaogan26'
    rootDir = 'images/' + cla + '/'
    save_dir_A = 'yaogan_many/image_after/'
    save_dir_B = 'yaogan_many/image_before/'
    save_dir_C = 'yaogan_many/image_jitter/'
    
    
    if not os.path.exists(save_dir_A):
        os.makedirs(save_dir_A)
        os.makedirs(save_dir_B)
        os.makedirs(save_dir_C)
        
    #list_dirs = os.walk(rootDir) 
    #image_list,name = list_image_files(rootDir)
    
    img_name = 'yaogan26\\m_airport_raw.png'
    img_raw = load_image(img_name)
    img_raw = preprocess(img_raw)
    
    img_name = 'yaogan26\\m_airpot_deform.png'
    img_deform = load_image(img_name)
    img_deform = preprocess(img_deform)
    
    
    # train dataset
    save_num = 200
    for i in range(save_num):
        x_index = randint(10,400)
        y_index = randint(10,600)
        subimg_raw = img_raw[x_index:x_index+256, y_index:y_index+256]
        subimg_deform, jit = jitter2D_single(subimg_raw)
        subimg_deform = img_deform[x_index:x_index+256, y_index:y_index+256]
        save_path_A = os.path.join(save_dir_A, str(i)+'.png')
        save_path_B = os.path.join(save_dir_B, str(i)+'.png')
        save_path_C = os.path.join(save_dir_C, str(i)+'.npy')
        
        subimg_raw_u = subimg_raw * 255.
        subimg_raw_u = subimg_raw_u.astype(np.uint8)
    
        subimg_deform_u = subimg_deform * 255.
        subimg_deform_u = subimg_deform_u.astype(np.uint8)
    
        Im = Image.fromarray(subimg_raw_u)
        Im.save(save_path_B)  
    
        Im = Image.fromarray(subimg_deform_u)
        Im.save(save_path_A)  
        
        np.save(save_path_C, jit) 
if __name__== '__main__':
    img_name = 'yaogan26\\m_airport_raw.png'
    img_raw = load_image(img_name)
    img_raw = preprocess(img_raw)
    
    img_name = 'yaogan26\\m_airpot_deform.png'
    img_deform = load_image(img_name)
    img_deform = preprocess(img_deform)
    x_index = 170
    y_index = 370
    img_raw_sample = img_raw[x_index:x_index+256, y_index:y_index+256]
    jix = np.load('yaogan26\\air_jix.npy')
    plt.imshow(img_raw_sample, cmap='gray')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    