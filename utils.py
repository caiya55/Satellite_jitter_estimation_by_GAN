import os
from PIL import Image
import numpy as np


RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.npy']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img

def load_image_with_C(path):
    img = np.load(path)
    return img
def preprocess_image(cv_img):
#    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
#    img = (img ) / 255
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images, istrain = True):
    sub = True
    images_A, images_B, images_C = [], [], []
    images_A_paths, images_B_paths, images_C_paths = [], [], []
    
    for root, dirs, files in os.walk(path):  
          if sub:
              for classes in dirs:
                  path_class = os.path.join(root,classes)
                  if istrain:
                      A_paths, B_paths, C_paths = os.path.join(path_class, 'A_train'), os.path.join(path_class, 'B_train'), os.path.join(path_class, 'C_train')
                  else:
                      A_paths, B_paths, C_paths = os.path.join(path_class, 'A_test'), os.path.join(path_class, 'B_test'), os.path.join(path_class, 'C_test')
                  all_A_paths, all_B_paths, all_C_paths = list_image_files(A_paths), list_image_files(B_paths), list_image_files(C_paths)
                  
                  for path_A, path_B, path_C, i in zip(all_A_paths, all_B_paths, all_C_paths, range(len(all_A_paths))):
                        img_A, img_B, img_C = load_image(path_A), load_image(path_B), load_image_with_C(path_C)
                        images_A.append(preprocess_image(img_A))
                        images_B.append(preprocess_image(img_B))
                        images_C.append((img_C))
                        images_A_paths.append(path_A)
                        images_B_paths.append(path_B)
                        images_C_paths.append(path_C)
                        if i+1 >= n_images : break
          sub = False
    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths),        
        'C': np.array(images_C),
        'C_paths': np.array(images_C_paths)
    }
    
def load_images_with_C(path, n_images, istrain = True):
    sub = True
    images_A, images_C = [], []
    images_A_paths, images_C_paths = [], []
    
    for root, dirs, files in os.walk(path):  
          if sub:
              for classes in dirs:
                  path_class = os.path.join(root,classes)
                  if istrain:
                      A_paths, C_paths = os.path.join(path_class, 'A_train'), os.path.join(path_class, 'C_train')
                  else:
                      A_paths, C_paths = os.path.join(path_class, 'A_test'), os.path.join(path_class, 'C_test')
                  all_A_paths, all_C_paths = list_image_files(A_paths), list_image_files(C_paths)
                  
                  for path_A, path_C, i in zip(all_A_paths, all_C_paths, range(len(all_A_paths))):
                        img_A, img_C = load_image(path_A), load_image_with_C(path_C)
                        images_A.append(preprocess_image(img_A))
                        images_C.append((img_C))
                        images_A_paths.append(path_A)
                        images_C_paths.append(path_C)
                        if i+1 >= n_images : break
          sub = False
    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'C': np.array(images_C),
        'C_paths': np.array(images_C_paths)
    }
    
def load__class_images(path, n_images, istrain = False):
    if istrain:
        A_paths, B_paths = os.path.join(path, 'A_train'), os.path.join(path, 'B_train')
    else:
        A_paths, B_paths = os.path.join(path, 'A_test'), os.path.join(path, 'B_test')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }
    
def load__class_images_with_C(path, n_images, istrain = False):
    if istrain:
        A_paths, B_paths, C_paths = os.path.join(path, 'A_train'), os.path.join(path, 'B_train'), os.path.join(path, 'C_train')
    else:
        A_paths, B_paths, C_paths = os.path.join(path, 'A_test'), os.path.join(path, 'B_test'), os.path.join(path, 'C_test')
    all_A_paths, all_B_paths, all_C_paths = list_image_files(A_paths), list_image_files(B_paths), list_image_files(C_paths)
    images_A, images_B, images_C = [], [], []
    images_A_paths, images_B_paths, images_C_paths = [], [], []
    for path_A, path_B, path_C in zip(all_A_paths, all_B_paths, all_C_paths):
        img_A, img_B, img_C = load_image(path_A), load_image(path_B), load_image_with_C(path_C)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_C.append((img_C))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        images_C_paths.append(path_C)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths),        
        'C': np.array(images_C),
        'C_paths': np.array(images_C_paths)
    }

def load__attitude_images(path, n_images):
    A_paths, B_paths = os.path.join(path, 'A_test_attitude'), os.path.join(path, 'B_test_attitude')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }   
def load_jitter(path, n_images):
    path = os.path.join(path, 'jitter')
    image_list = list_image_files(path)
    jitter_all = []
    for path in image_list:
        jitter = np.load(path)
        jitter_all.append(jitter)
        if len(jitter_all) > n_images - 1:break
    return jitter_all
if __name__ == '__main__':
    path = '..\dataset\image_deform\\'
    n_images = 10
    sub = True
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    
    for root, dirs, files in os.walk(path):  
          if sub:
              for classes in dirs:
                  path_class = os.path.join(root,classes)
                  A_paths, B_paths = os.path.join(path_class, 'A'), os.path.join(path_class, 'B')
                  all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
                  
                  for path_A, path_B, i in zip(all_A_paths, all_B_paths, range(len(all_A_paths))):
                        img_A, img_B = load_image(path_A), load_image(path_B)
                        images_A.append(preprocess_image(img_A))
                        images_B.append(preprocess_image(img_B))
                        images_A_paths.append(path_A)
                        images_B_paths.append(path_B)
                        if i+1 >= n_images : break
          sub = False