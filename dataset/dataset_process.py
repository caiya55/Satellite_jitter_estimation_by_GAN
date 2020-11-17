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
    img = img.resize((256, 256))  
    # low down the resolution of the image
    return img
def preprocess(img):
    img = np.array(img) 
    img = img / 255
#    img = (img - 127.5) / 127.5
    return img
def jitter():
    x = np.arange(width)
#    f=[0.35,0.5, 0.8,1];
    f=[0.1, 0.2, 0.3, 0.4];    
    rf = random() + 0.8
    f = [f*rf for f in f]   #fre is 0.05-0.2
    #pha = [0.5,0.6,0.7,0.8];
    pha = random_sample(4)*6.28
    
    amp = [1, 0.2, 0.1, 0.05]
    ra = random()*2
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



# main loop 
#classes = obtain_classes()  
#for cla in classes:
cla = 'freeway'
rootDir = 'images/' + cla + '/'
save_dir_A = 'image_deform/' + cla + '/A_train/'
save_dir_B = 'image_deform/' + cla + '/B_train/'
save_dir_C = 'image_deform/' + cla + '/C_train/'

save_dir_A_test = 'image_deform/' + cla + '/A_test/'
save_dir_B_test = 'image_deform/' + cla + '/B_test/'
save_dir_C_test = 'image_deform/' + cla + '/C_test/'

if not os.path.exists(save_dir_A):
    os.makedirs(save_dir_A)
    os.makedirs(save_dir_B)
    os.makedirs(save_dir_C)
    os.makedirs(save_dir_C_test)
    os.makedirs(save_dir_A_test)
    os.makedirs(save_dir_B_test)
    
list_dirs = os.walk(rootDir) 
image_list,name = list_image_files(rootDir)

for i in range(len(image_list)//2):  # all the images
    print(i)
    img_gray = load_image(image_list[i])
    img = preprocess(img_gray)
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    f = interp2d(x, y, img, kind='linear')
    jix = jitter()
    # make a loop
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        out_tmp = f(y+jix[index], x[index]).T
        img_out[index] = out_tmp
    img_out = img_out * 255.
    img_out = img_out.astype(np.uint8)
    
    # output the jitter map
    jix = np.reshape(jix,[256,1])
    ones = np.ones([1,256])
#    jitter_map = np.dot(jix,ones )
    jitter_map = jix

    # save the image
    save_path_A = os.path.join(save_dir_A, name[i])
    save_path_B = os.path.join(save_dir_B, name[i])
    save_path_C = os.path.join(save_dir_C, name[i])+'.npy'
    Im = Image.fromarray(img_out)
    
    Im.save(save_path_A)  
    img_gray.save(save_path_B)  
    np.save(save_path_C, jitter_map)
for i in range(len(image_list)//2, len(image_list)):  # all the images
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
    
    # output the jitter map
    jix = np.reshape(jix,[256,1])
    ones = np.ones([1,256])
#    jitter_map = np.dot(jix,ones )
    jitter_map = jix
    # save the image
    save_path_A = os.path.join(save_dir_A_test, name[i])
    save_path_B = os.path.join(save_dir_B_test, name[i])
    save_path_C = os.path.join(save_dir_C_test, name[i])+'.npy'
    Im = Image.fromarray(img_out)
    
    Im.save(save_path_A)  
    img_gray.save(save_path_B)  
    np.save(save_path_C, jitter_map)
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