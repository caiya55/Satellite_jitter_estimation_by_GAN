# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:04:46 2018

@author: zhang
"""
'''load the pretrained model and output the jitter curves, for major revision'''
import tensorflow as tf
from model_TF import D_on_G
from config import get_config
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from dataset_process_attitude import  jitter2D
from random import random
from numpy.random import random_sample
from numpy import sin
from scipy.interpolate import interp2d
import scipy.io as scio

#import pylab as pl


#pl.style.use('ggplot')

def jitter(width):
    x = np.arange(width)
    f=[0.1, -0.05, 0.02, 0.01];    
    rf = 0.5 + random()*0.8
    f = [f*rf for f in f]
    pha = random_sample(4)*6.28
    amp = [1, 0.2, 0.05, 0.01]  
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
    
def jitter_reverse(jix_test,jiy_test):
    line = range(0,296)   
    change_line_x = jix_test
    change_line_y = jiy_test + line
    jix_new = np.zeros(256)
    jiy_new = np.zeros(256)
    for i in range(20, 276):
        cand, jix_tmp, a1 = find_min(i, change_line_y[i-20:i+20], change_line_x[i-20:i+20])
        jiy_new[i-20] = cand -20    
        jix_new[i-20] = -jix_tmp
    return jix_new, jiy_new

def jitters_reverse(jit_in):
    '''specifise the size of the input jitter is 296 and inter is 10'''
    jitt_rev = np.zeros([256,6])
    jitt_rev[:,0], jitt_rev[:,1] = jitter_reverse(jit[:296,0], jit[:296,1])
    jitt_rev[:,2], jitt_rev[:,3] = jitter_reverse(jit[10:296+10,0], jit[10:296+10,1])
    jitt_rev[:,4], jitt_rev[:,5] = jitter_reverse(jit[20:296+20,0], jit[20:296+20,1])
    return jitt_rev
def uint_img(img):
    return (img*127.5+127.5).astype(np.uint8)

def img_diff(img1, img2):
    dif1 = (img1- img2)*127.5 + 100
    
    plt.imshow(dif1, cmap='gray')
    plt.grid(False)
    plt.axis('off')
    plt.show()
    return dif1

def load_image(image_name):
    img = Image.open(image_name)
    img = img.resize((256+40, 256+40)) 
    img = np.array(img)
    img = img / 255
    return img  

def image_deformed(img_name, inter):
    img = load_image(img_name)
    width = img.shape[0]

    x = np.arange(width).astype(float)
    y = np.arange(width).astype(float)
    '''create the jitter '''
    jix = jitter(width + inter*2)#roll
    factor = 0.3 + random()*0.4
    jiy = factor*jitter(width + inter*2)# pitch
    # make a loop
    img_out = np.zeros(img.shape)
    for i in range(3):
        '''create the interpolation'''
        img_gray = img[:,:,i]
        f = interp2d(x, y, img_gray, kind='linear')
        jix_gray = jix[i*inter: i*inter+width]
        jiy_gray = jiy[i*inter: i*inter+width]
        img_out_gray = np.zeros(img_gray.shape)
        for index in range(img_gray.shape[0]):
            '''x is x direction, y is y direction'''
            out_tmp = f(x+jix_gray[index], y[index]+jiy_gray[index]).T # jix is 
            img_out_gray[index] = out_tmp
        img_out_gray = img_out_gray * 255.
        img_out_gray = img_out_gray.astype(np.uint8)
        img_out[:,:,i] = img_out_gray
    jit = np.concatenate([jix[...,None],jiy[...,None]],axis=-1)
    return img_out.astype(np.uint8),(img*255).astype(np.uint8), jit
def preprocess(img_out, img_raw, jit_in):
    img_out = (img_out - 127.5) / 127.5
    img_raw = (img_raw - 127.5) / 127.5
    jit_rev = jitters_reverse(jit_in)
    batch_x = np.zeros([5,256,256,1])
    batch_y = np.zeros([5,256,256,1])
    batch_z = np.zeros([5,256,2])
    for i in range(5):
        batch_x[i,:,:,0] = img_out[20:276,20:276,0]
    for i in range(5):
        batch_y[i,:,:,0] = img_raw[20:276,20:276,0]
    for i in range(5):
        batch_z[i,:,0] = jit_rev[:,0] # first image x
        batch_z[i,:,1] = jit_rev[:,1] # first image y
    return batch_x, batch_y, batch_z, jit_rev

parser = argparse.ArgumentParser()
parser.add_argument("--final_layer", type=int, help="choose the number of final layers", default = 128)
parser.add_argument("--alpha", type=float, help="choose the value of alpha", default = 1)
parser.add_argument("--max_pooling", type=bool, help="choose whether max_pooling is used", default = True)
parser.add_argument("--kernel_size", type=int, help="choose size of the kernel", default = 3)

args = parser.parse_args()

if __name__=='__main__':
    sess = tf.Session()
    config = get_config(is_train=True)
    
    restore = D_on_G(sess, config, "DIRNet", args, is_train=True)
    restore.restore(config.ckpt_dir)
    
    '''unique image preprocess and jitter creation strategy'''
    img_name = 'resultsforIGARSS\\solarpanel660.jpg'
    img_out, img_raw, jit = image_deformed(img_name, 10)
    '''here img_out and img_raw is 296*296*3 image uint8, jit is (296+20)*2'''
    batch_x, batch_y, batch_z, jitt_rev = preprocess(img_out, img_raw, jit)
    '''save the img out deformed colored image'''
    cv2.imwrite('resultsforIGARSS//deform_test.jpg', img_out)
    cv2.imwrite('resultsforIGARSS//'+img_name+'1.png',uint_img(batch_y[0,:,:,0]))
    loss1, loss2, output,wrap_yaogan = restore.predict_one(batch_x[0], config)        
    '''show and save the images'''
    plt.imshow(output[:,:,0], cmap='gray')
    plt.grid(False)
    plt.axis('off')   
    plt.title('Rstored image')
    plt.show()            
    
    plt.imshow(batch_x[0,:,:,0], cmap='gray')
    plt.grid(False)
    plt.axis('off')   
    plt.title('Raw image')
    plt.show()   
       
    '''find the difference of the images'''
    '''tempoararly'''

    diff_dt = img_diff(batch_x[0,:,:,0], batch_y[0,:,:,0] )
    diff_rt = img_diff(output[:,:,0], batch_y[0,:,:,0])
    
        
    plt.plot(wrap_yaogan[:,0])
    plt.plot(wrap_yaogan[:,1])
    plt.plot(batch_z[0,:,0])
    plt.plot(batch_z[0,:,1])
    plt.grid(True)
    plt.xlabel('Lines')
    plt.ylabel('Pixel')
    plt.legend(['Restored cross-track', 'Restored along-track', 'Raw cross-track', 
                'Raw along-track'], ncol=2)
    
    '''save all the information'''
#    plt.savefig('resultsforIGARSS//'+img_name[:-1]+'_jit.png')
#    plt.show()
    '''save the imfor in to mat, so matlab can process them'''
    data_re = {'A': jitt_rev}
    scio.savemat('resultsforIGARSS//jitter_rev.mat', {'A':data_re['A']})
    data_re = {'A': wrap_yaogan}
    scio.savemat('resultsforIGARSS//wrap_yaogan.mat', {'A':data_re['A']})

#    
    cv2.imwrite('resultsforIGARSS//out-truth1.png', img_raw[20:276,20:276,:])
    cv2.imwrite('resultsforIGARSS//out-deform1.png', img_out[20:276,20:276,:])
#    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'2.png', uint_img(batch_x[0,:,:,0]))
#    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'diff-d-t.png', (diff_dt).astype(np.uint8))
#    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'diff-r-t.png', (diff_rt).astype(np.uint8))
#    '''save the jitter'''
#    jit_name = 'resultsforIGARSS//'+img_name + '.npy'
#    np.save(jit_name, [wrap_yaogan,batch_z[0,:]])
