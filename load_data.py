from glob import glob
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

def load_train_data(path_train, is_normalize = True):
    images = []
    masks = []
    for path in glob(path_train):
        # load image
        img_path = glob(path + '/images/*.png')[0] # 1 image per folder
        masks_path = glob(path + '/masks/*.png') # a list of masks per folder
        image = cv2.imread(img_path)[:,:,:3] # 4D to 3D
        # load mask
        mask = np.zeros(imread(masks_path[0]).shape)
        mask_r = np.expand_dims(mask, axis=2)
        for mask_path in masks_path:
            temp = cv2.imread(mask_path, 0) # 2D image
            temp_r = np.expand_dims(temp, axis=2)
            mask_r += temp_r
        mask_r[mask_r>255] = 255
        images.append(image)
        masks.append(mask_r)

    if is_normalize:
        images = scale_normalize_image(images, h=256, w=256, c=3)
        masks = scale_normalize_image(masks, h=256, w=256, c=1)

    images, masks = np.stack(images, axis=0), np.stack(masks, axis=0)
    return images, masks

    
def load_test_data(path_test, is_normalize = True):
    images = []
    for path in glob(path_test):
        # load image
        img_path = glob(path + '/images/*.png')[0] # 1 image per folder
        image = imread(img_path)[:,:,:3] # 4D to 3D
        images.append(image)
        
    if is_normalize:
        images = scale_normalize_image(images, h=256, w=256, c=3)
        images = np.stack(images, axis=0)
        
    return images

# scale images and masks to (255, 255, 3)
def scale_normalize_image(images, h=256, w=256, c=3):
    image_rescaled = []
    for i in images:
        i = i.astype(np.float32)
        i /= 255.
        img = resize(i, (h,w), c, mode='constant', preserve_range=True)
        image_rescaled.append(img)
    return image_rescaled




def main():
    path_train = "./data/stage1_train/*"
    path_test = "./data/stage1_test/*"
    train_images, train_masks = load_train_data(path_train, True)
    test_images = load_test_data(path_test, True)
    test_images_ori = load_test_data(path_test, False)

    # np.save("train_image_small",train_images[0:60])
    # np.save("train_mask_small",train_masks[0:60])

    np.save("train_image",train_images)
    np.save("train_mask",train_masks)
    np.save("test_image",test_images)
    np.save("test_image_ori",test_images_ori)

if __name__ == '__main__':
    main()
