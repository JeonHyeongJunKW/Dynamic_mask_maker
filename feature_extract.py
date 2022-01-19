import cv2
import numpy as np

def extract_SIFT(img, x,y, pixel_size):
    detector = cv2.xfeatures2d.SIFT_create(1)
    kp, des = detector.detectAndCompute(img, None)
    return