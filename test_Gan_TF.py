# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 15:04:46 2018

@author: zhang
"""
'''load the pretrained model and output the jitter curves, for major revision'''
import tensorflow as tf
from model_TF import D_on_G
from config import get_config
#from data import MNISTDataHandler
from ops import mkdir
from keras.models import load_model
from utils import load_images, load_images_with_C
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse
from dataset_process_attitude import  jitter2D
import pylab as pl


#pl.style.use('ggplot')

def load_image(path):
    img = Image.open(path)#.convert('L')
    return img


def preprocess(path, x_b, y_b):
    img = load_image(path)
    img = img.resize((256+40, 256+40)) 
    img = np.array(img)
    img = img[:,:,0]
#    img = img[x_b:x_b+256, y_b:y_b+256]
    img = (img - 127.5) / 127.5
    num = 5
    out = np.zeros([5,img.shape[1],img.shape[1],1])
    for i in range(num):
        out[i,:,:,0] = img
    return out
n_images = 400

def uint_img(img):
    return (img*127.5+127.5).astype(np.uint8)

def img_diff(img1, img2):
    dif1 = (img1- img2)*127.5 + 100
    
    plt.imshow(dif1, cmap='gray')
    plt.grid(False)
    plt.axis('off')
    plt.show()
    return dif1

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
    
    img_name = 'solarpanel660'
    batch_y = preprocess('resultsforIGARSS//' + img_name+'.jpg', 0, 0)

    batch_x, batch_z = jitter2D(batch_y)  
    if batch_x.shape[1] == 296:
        bo=20
        batch_x = batch_x[:,bo:bo+256,bo:bo+256,:] 
        batch_y = batch_y[:,bo:bo+256,bo:bo+256,:] 
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
    plt.savefig('resultsforIGARSS//'+img_name[:-1]+'_jit.png')
    plt.show()
    
    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'1.png', uint_img(output[:,:,0]))
    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'2.png', uint_img(batch_x[0,:,:,0]))
    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'diff-d-t.png', (diff_dt).astype(np.uint8))
    cv2.imwrite('resultsforIGARSS//'+img_name[:-1]+'diff-r-t.png', (diff_rt).astype(np.uint8))
    '''save the jitter'''
    jit_name = 'resultsforIGARSS//'+img_name + '.npy'
    np.save(jit_name, [wrap_yaogan,batch_z[0,:]])
