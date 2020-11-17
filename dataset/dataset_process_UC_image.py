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
import copy
import tifffile as tiff
import cv2
# data：2018/5/27
PATH = 'UCMerced_LandUse/Images/'
CLA = 'runway'
'''
jix is jitter in x direction, jiyy is jitter in y direction
'''
def obtain_classes():
    path = PATH
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
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.tif']
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
#    img = Image.open(path).convert('L')
    img = tiff.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256+bo, 256+bo))  
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
    f=[0.1, -0.05, 0.02, 0.01];    
    rf = 0.6 + random()*0.3
#    rf = 0.3 + random()*0.3

    f = [f*rf for f in f]
    pha = random_sample(4)*6.28
        
    amp = [1, 0.1, 0.05, 0.01]  
#    ra = 8 + random()*2
    ra = 4 + random()*3
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
    
def jitter_reverse(jitter):
    line = range(0,296)   
    jix_test = jitter[:,0]
    jiy_test = jitter[:,1]
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
        cand, jix_tmp, a1 = find_min(i, change_line_y[i-15:i+15], change_line_x[i-15:i+15])
        jiy_new[i-20] = cand + 5 - 20    
        jix_new[i-20] = -jix_tmp
    return jix_new, jiy_new, error



# main loop 
#classes = obtain_classes()  
#for cla in classes:
cla = CLA
rootDir = PATH + cla + '/'
save_dir_A = 'yaogan_many/' + cla + '/A_train/'
save_dir_B = 'yaogan_many/' + cla + '/B_train/'
save_dir_C = 'yaogan_many/' + cla + '/C_train/'

save_dir_A_test = 'yaogan_many/' + cla + '/A_test/'
save_dir_B_test = 'yaogan_many/' + cla + '/B_test/'
save_dir_C_test = 'yaogan_many/' + cla + '/C_test/'

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

for i in range(int(len(image_list)/2)):  # all the images
    print(i)
    img_gray = load_image(image_list[i])
    img, img_gray_save = preprocess(img_gray)
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    f = interp2d(x, y, img, kind='linear')
#    jix = jitter(width)
    jix = jitter(width) # pitch
#    jiy = copy.deepcopy(jix)
#    factor = 0.6 + random()*0.4
#    jiy = factor*jiy  # roll
    jiy = jitter(width)

    # make a loop
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        '''x is x direction, y is y direction'''
        out_tmp = f(x+jix[index], y[index]+jiy[index]).T # jix is 
        img_out[index] = out_tmp
    img_out = img_out * 255.
    img_out = img_out.astype(np.uint8)
    
    # output the jitter map
#    jix = np.reshape(jix,[256,1])
    jitter_map = np.zeros([256+bo,2])
    jitter_map[:,0] = jix
    jitter_map[:,1] = jiy
    
    jix_r, jiy_r, _ = jitter_reverse(jitter_map)
    ji_out = np.zeros([256,2])
    ji_out[:,0] = jix_r
    ji_out[:,1] = jiy_r
#    print('max-',jix_r.max(),' may-',jiy_r.max() )
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
    np.save(save_path_C, ji_out)
    
for i in range(int(len(image_list)/2),len(image_list)):  # all the images
    print(i)
    img_gray = load_image(image_list[i])
    img, img_gray_save = preprocess(img_gray)
    width = img.shape[0]
    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    
    f = interp2d(x, y, img, kind='linear')
#    jix = jitter(width)
    jix = jitter(width)
#    jiy = copy.deepcopy(jiy)
#    jiy = 0.75*jiy
    jiy = jitter(width)
    # make a loop
    img_out = np.zeros(img.shape)
    for index in range(img.shape[0]):
        out_tmp = f(x+jix[index], y[index]+jiy[index]).T
        img_out[index] = out_tmp
    img_out = img_out * 255.
    img_out = img_out.astype(np.uint8)
#    plt.imshow(img_out, cmap = 'gray')
#    plt.show()
    
    
    # output the jitter map
#    jix = np.reshape(jix,[256,1])
    jitter_map = np.zeros([256+bo,2])
    jitter_map[:,0] = jix
    jitter_map[:,1] = jiy
    jix_r, jiy_r, _ = jitter_reverse(jitter_map) # x is roll, y is pitch
    ji_out = np.zeros([256,2])
    ji_out[:,0] = jix_r
    ji_out[:,1] = jiy_r
#    plt.plot(jix_r)
#    plt.plot(jiy_r)
#    plt.show()
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
    np.save(save_path_C, ji_out)

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