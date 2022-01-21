import cv2
import numpy as np
from dsift import SingleSiftExtractor

def extract_SIFT(img, x,y, pixel_size):
    bottom_x = int(x-pixel_size/2) if int(x-pixel_size/2) >=0 else 0
    bottom_y = int(y - pixel_size / 2) if int(y - pixel_size / 2) >= 0 else 0
    top_x = int(x + pixel_size / 2) if int(x + pixel_size / 2) < img.shape[1] else img.shape[1]-1
    top_y = int(y + pixel_size / 2) if int(y + pixel_size / 2) < img.shape[0] else img.shape[0]-1
    patch_img = img[bottom_y:top_y+1,bottom_x:top_x+1]
    extractor = SingleSiftExtractor(pixel_size)
    dense_feature = extractor.process_image(patch_img)
    return dense_feature

def Get_Fundamental(img1, img2):
    descriptor = cv2.ORB_create(2000)
    kp1,des1 = descriptor.detectAndCompute(img1,None)
    kp2,des2 = descriptor.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    pts1 =[]
    pts2 =[]
    for i in range(len(matches)):
        pts1.append(kp1[matches[i].queryIdx].pt)
        pts2.append(kp2[matches[i].trainIdx].pt)


        if i==1000:
            break
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    return F