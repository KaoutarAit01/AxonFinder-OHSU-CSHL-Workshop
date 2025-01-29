import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import datapreparation.tiling_images
from tqdm import tqdm

%env SM_FRAMEWORK=tf.keras
from model.buildmodel import build_model

model = build_model()

path = "../model_checkpoints/04112023195459/best_model.ckpt"

model.load_weights(path)

images_path = 'path-images'
save_path = 'path-to-save-masks'

# load image
images_name = [name for name in os.listdir(images_path) if name.endswith(".png")]

for i in range(len(images_name)):
    image = cv2.imread(os.path.join(images_path, images_name[i]))
    image = np.moveaxis(image, 2, 0)
    
    # tile image
    t = tiling_images.tile_image(image, tile_size=512, p_overlap=0.15)
    tiles = np.array(t['image'])
    tiles = np.moveaxis(tiles, 1, 3)
    masks = model.predict(tiles/255)
    
    masks = np.argmax(masks, axis=-1)
    
    t['mask'] = masks
    
    _, M = tiling_images.stitch_image(t['tile_size'], t)
    M = M*255
        
    cv2.imwrite(os.path.join(save_path, images_name[i]), M)
