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
    f=[0.1, -0.05, 0.02, 0.01];    
    rf = 0.5 + random()*0.8 

    f = [f*rf for f in f]
    pha = random_sample(4)*6.28
        
    amp = [1, 0.2, 0.05, 0.01]  
#    ra = 8 + random()*2
    ra = 2 + random()*2
    amp = [amp*ra for amp in amp]
    jix  = amp[0] * sin(f[0] * x + pha[0]) + amp[1] * sin(f[1] * x+pha[1]) + \
                    amp[2] * sin(f[2] * x + pha[2]) + amp[3] * sin(f[3] * x + pha[3]);
    return jix


def find_min(idx, ac, acx):
    
    mam =  np.argmin(abs(ac-idx))
    jix = acx[mam]
    a0 = idx - ac[mam-1]
    a1 = idx - ac[mam]
    a2 = idx - ac[mam+1]
    jix_b = acx[mam-1]
    jix_a = acx[mam+1]
#    return round(mam), jix, a1

    if a1*a2 < 0:
        decimal = abs(a1)/(abs(a1) + abs(a2))
        jix_out = decimal*jix + (1 - decimal)*jix_a
        return ((mam + decimal)), jix_out,  a1
    else:
        decimal = abs(a1)/(abs(a1) + abs(a0))
        jix_out = decimal*jix + (1 - decimal)*jix_b
        
        return ((mam - decimal)), jix_out, a1

def jitter_reverse(jix_test, jiy_test):
    line = range(0,296)   
    change_line_x = jix_test
    change_line_y = jiy_test + line
    jix_new = np.zeros(256)
    jiy_new = np.zeros(256)
    error = np.zeros(256)
#    for i in range(20, 276):
#        cand, a1 = find_min(i, change_line_x[i-15:i+15])
#        jix_new[i-20] = cand + 5 - 20
#        error[i-20] = a1 
    for i in range(20, 276):
        cand, jix_tmp, a1 = find_min(i, change_line_y[i-20:i+20], change_line_x[i-20:i+20])
        jiy_new[i-20] = cand -20    
        jix_new[i-20] = -jix_tmp
    return jix_new, jiy_new, error

def jitter2D(imgs):
    imgs = imgs[:,:,:,0]
    width = imgs.shape[1]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    imgs_out = np.zeros_like(imgs)
    jix_out_x = np.zeros([imgs.shape[0], 256])
    jix_out_y = np.zeros([imgs.shape[0], 256])
    for i in range(imgs.shape[0]):
        jix = jitter(width) 
        factor = 0.3 + random()*0.2   
        jiy = factor*jitter(width) 
        f = interp2d(x, y, imgs[i], kind='linear')
        for index in range(width):
            out_tmp = f(y+jix[index], x[index]+jiy[index]).T
            imgs_out[i, index] = out_tmp
        jix_r, jiy_r, _ = jitter_reverse(jix, jiy)
        jix_out_x[i] = jix_r
        jix_out_y[i] = jiy_r
    imgs_out = imgs_out[:,:,:,np.newaxis]   
    jix_out_x = jix_out_x[:,:,np.newaxis]
    jix_out_y = jix_out_y[:,:,np.newaxis]
    jix_out = np.concatenate([jix_out_x,jix_out_y],axis=-1)
    return imgs_out, jix_out

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