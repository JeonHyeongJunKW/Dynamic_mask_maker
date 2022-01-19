import cv2
import numpy as np
from dsift import SingleSiftExtractor

def extract_SIFT(img, x,y, pixel_size):
    global detector

    bottom_x = int(x-pixel_size/2) if int(x-pixel_size/2) >=0 else 0
    bottom_y = int(y - pixel_size / 2) if int(y - pixel_size / 2) >= 0 else 0
    top_x = int(x + pixel_size / 2) if int(x + pixel_size / 2) < img.shape[1] else img.shape[1]-1
    top_y = int(y + pixel_size / 2) if int(y + pixel_size / 2) < img.shape[0] else img.shape[0]-1
    patch_img = img[bottom_y:top_y+1,bottom_x:top_x+1]
    extractor = SingleSiftExtractor(pixel_size)
    dense_feature = extractor.process_image(patch_img)
    return dense_feature