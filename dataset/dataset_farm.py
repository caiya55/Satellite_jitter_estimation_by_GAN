# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:16:43 2018

@author: zhang
"""

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
    path = 'images_farm/'
    sub = True
    classes = []
    for root, dirs, files in os.walk(path):
        if sub:
            for cla in dirs:
                classes.append(cla)
        sub = False        
    return classes

flip = 1
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
bo = 40
def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((256+bo, 256+bo))  
    # low down the resolution of the image
    return img
def preprocess(img):
    img = np.array(img) 
    if flip == 0:
        img = np.rot90(img,2)
    Im_save = Image.fromarray(img)
    img = img / 255
#    img = (img - 127.5) / 127.5
    return img, Im_save
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
    mu = 4
    amp[0] = (random()*(-2) + 1)*2
    amp[1] = (random()*(-2) + 1)*2
    amp[2] = (random()*(-2) + 1)*2
    amp[3] = (random()*(-2) + 1)*2


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
    ra = random()*0.2 + 0.3 
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




# main loop 
#classes = obtain_classes()  
#for cla in classes:
cla = 'farms'
rootDir = 'images_farm/' + cla + '/'
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

#img_gray = load_image(image_list[0])
#img = preprocess(img_gray)
#
#arr = img * 255.
#arr = arr.astype(np.uint8)
#plt.imshow(arr,cmap='gray')
#plt.show()

for i in range(len(image_list)):  # all the images
    print(i)
    img_gray = load_image(image_list[i])
    img, img_gray_save = preprocess(img_gray)
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    f = interp2d(x, y, img, kind='linear')
#    jix = jitter(width)
    jix = jitter(width)  # roll
    jiy = jitter_y(width) # pitch 
    # make a loop
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        out_tmp = f(y+jix[index], x[index]+jiy[index]).T
        img_out[index] = out_tmp
    img_out = img_out * 255.
    img_out = img_out.astype(np.uint8)
    
    # output the jitter map
#    jix = np.reshape(jix,[256,1])
    jitter_map = np.zeros([256+bo,2])
    jitter_map[:,0] = jix
    jitter_map[:,1] = jiy
    
    if flip == 0:
        name_out = str(i) + name[0][-4:]
    else:
        name_out = str(i) + '_1' + name[0][-4:]

    # save the image
    save_path_A = os.path.join(save_dir_A, name_out)
    save_path_B = os.path.join(save_dir_B, name_out)
    save_path_C = os.path.join(save_dir_C, name_out)+'.npy'
    Im = Image.fromarray(img_out)
    
    Im.save(save_path_A)  
    img_gray_save.save(save_path_B)  
    np.save(save_path_C, jitter_map)
    
for i in range(len(image_list)):  # all the images
    print(i)
    img_gray = load_image(image_list[i])
    img, img_gray_save = preprocess(img_gray)
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    f = interp2d(x, y, img, kind='linear')
#    jix = jitter(width)
    jix = jitter(width)
    jiy = jitter_y(width)
    # make a loop
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        out_tmp = f(y+jix[index], x[index]+jiy[index]).T
        img_out[index] = out_tmp
    img_out = img_out * 255.
    img_out = img_out.astype(np.uint8)
    
    # output the jitter map
#    jix = np.reshape(jix,[256,1])
    jitter_map = np.zeros([256+bo,2])
    jitter_map[:,0] = jix
    jitter_map[:,1] = jiy
#    jitter_map = np.dot(jix,ones )
#    jitter_map = jix
    # save the image
    if flip == 0:
        name_out = str(i) + name[0][-4:]
    else:
        name_out = str(i) + '_1' + name[0][-4:]

    save_path_A = os.path.join(save_dir_A_test, name_out)
    save_path_B = os.path.join(save_dir_B_test, name_out)
    save_path_C = os.path.join(save_dir_C_test, name_out)+'.npy'
    Im = Image.fromarray(img_out)
    
    Im.save(save_path_A)  
    img_gray_save.save(save_path_B)  
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