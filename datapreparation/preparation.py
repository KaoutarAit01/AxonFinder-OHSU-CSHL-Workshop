import cv2
import numpy as np
import os
import tiling_images
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def augmentation_rotation(images, masks):
    rotation_img = []
    rotation_msk = []
    for i in range(len(images)):
        rotation_img.append(cv2.rotate(images[i], cv2.ROTATE_90_CLOCKWISE))
        rotation_img.append(cv2.rotate(images[i], cv2.ROTATE_180))
        rotation_img.append(cv2.rotate(images[i], cv2.ROTATE_90_COUNTERCLOCKWISE))
        rotation_msk.append(cv2.rotate(masks[i], cv2.ROTATE_90_CLOCKWISE))
        rotation_msk.append(cv2.rotate(masks[i], cv2.ROTATE_180))
        rotation_msk.append(cv2.rotate(masks[i], cv2.ROTATE_90_COUNTERCLOCKWISE))
    return rotation_img, rotation_msk

def augmentation_flipping(images, masks):
    flipping_img = []
    flipping_msk = []
    for i in range(len(images)):
        flipping_img.append(cv2.flip(images[i], 0))
        flipping_img.append(cv2.flip(images[i], 1))
        flipping_msk.append(cv2.flip(masks[i], 0))
        flipping_msk.append(cv2.flip(masks[i], 1))
    return flipping_img, flipping_msk

M = np.float32([[1, 0.5, 0],
                [0.3, 1, 0],
                [0, 0, 1]])

def augmentation_shearing(images, masks, m=M):
    shearing_img = []
    shearing_msk = []
    for i in range(len(images)):
        rows, cols, dims = images[i].shape
        shearing_img.append(cv2.warpPerspective(images[i], M, (int(cols*1.5), int(rows*1.3))))
        shearing_msk.append(cv2.warpPerspective(masks[i], M, (int(cols*1.5), int(rows*1.3))))
    return shearing_img, shearing_msk


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
    
def remove_duplicates(images, masks):
    INDEX = []
    for i in range(len(images)-1):
        for j in range(i+1,len(images)):
            if is_similar(images[i], images[j]):
                INDEX.append(j)
    INDEX = list(set(INDEX))
    INDEX.sort(reverse=True)
    for ind in INDEX:
        images.pop(ind)
        masks.pop(ind)
    return images, masks

f = np.array([[0, -1, 0], 
              [-1, 5, -1], 
              [0, -1, 0]]) # Applying cv2.filter2D function on image 

def augmentation_sharpness(images, masks, filter=f):
    sharpen_img = []
    for img in images:
        sharpen_img.append(cv2.filter2D(img, -1, filter))
    return sharpen_img, masks

def augmentation_brightness_contrast(image):
    alpha = np.random.randint(1, 8)
    beta = np.random.randint(0, 3)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def augmentation(images, masks):
    shimages, shmasks = augmentation_shearing(images, masks)
    rimages, rmasks = augmentation_rotation(images, masks)
    images += rimages
    masks += rmasks
    fimages, fmasks = augmentation_flipping(images, masks)
    images += fimages
    masks += fmasks
    images, masks = remove_duplicates(images, masks)
    simages, smasks = augmentation_sharpness(images, masks)
    images += simages + shimages
    masks += smasks + shmasks
    return images, masks


def Xy_preparation(images_path, masks_path, tile_size=512, p_split=0.2, p_overlap=0.15):
    
    masks = [os.path.join(masks_path, name) for name in sorted(os.listdir(masks_path)) if name.endswith(".png")]
    images = [os.path.join(images_path, name) for name in sorted(os.listdir(images_path)) if name.endswith(".png")]

    X = np.empty(shape=(1, 3, tile_size, tile_size), dtype=np.uint8)
    y = np.empty(shape=(1, tile_size, tile_size), dtype=np.uint8)

    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks[i], 0)
        image, mask = augmentation([image], [mask])
        image = [np.moveaxis(img, 2, 0) for img in image]
        for j in range(len(image)):
            X = np.concatenate([X, tiling_images.tile_image(image[j], tile_size=tile_size, p_overlap=p_overlap)['image']], axis=0)
            mask[j][mask[j] > 0] = 255
            y = np.concatenate([y, tiling_images.tile_image(mask[j], tile_size=tile_size, p_overlap=p_overlap)['image']], axis=0) 
        del image
        del mask
    
    X = X[1:]
    y = y[1:]
    X = np.moveaxis(X, 1, 3)
    indexes = np.array(list(set(np.arange(len(y))) - set(np.argwhere(y > 0)[:,0])))

    X = np.delete(X, indexes, axis=0)
    y = np.delete(y, indexes, axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_split, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test
