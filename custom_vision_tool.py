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

def extract_HoG(img,x,y,pixel_size):
    bottom_x = int(x - pixel_size / 2) if int(x - pixel_size / 2) >= 0 else 0
    bottom_y = int(y - pixel_size / 2) if int(y - pixel_size / 2) >= 0 else 0
    top_x = int(x + pixel_size / 2) if int(x + pixel_size / 2) < img.shape[1] else img.shape[1] - 1
    top_y = int(y + pixel_size / 2) if int(y + pixel_size / 2) < img.shape[0] else img.shape[0] - 1
    patch_img = img[bottom_y:top_y + 1, bottom_x:top_x + 1]
    hog = cv2.HOGDescriptor()
    descriptor = hog.compute(patch_img)
    return descriptor

def change_patch(source,s_y,s_x,target,t_y,t_x,pixel_size):
    s_bottom_x = int(s_x - pixel_size / 2) if int(s_x - pixel_size / 2) >= 0 else 0
    s_bottom_y = int(s_y - pixel_size / 2) if int(s_y - pixel_size / 2) >= 0 else 0
    s_top_x = int(s_x + pixel_size / 2) if int(s_x + pixel_size / 2) < target.shape[1] else target.shape[1] - 1
    s_top_y = int(s_y + pixel_size / 2) if int(s_y + pixel_size / 2) < target.shape[0] else target.shape[0] - 1

    t_bottom_x = int(t_x - pixel_size / 2) if int(t_x - pixel_size / 2) >= 0 else 0
    t_bottom_y = int(t_y - pixel_size / 2) if int(t_y - pixel_size / 2) >= 0 else 0
    t_top_x = int(t_x + pixel_size / 2) if int(t_x + pixel_size / 2) < target.shape[1] else target.shape[1] - 1
    t_top_y = int(t_y + pixel_size / 2) if int(t_y + pixel_size / 2) < target.shape[0] else target.shape[0] - 1

    # source_x_size = s_top_x- s_bottom_x
    # source_y_size = s_top_y - s_bottom_y
    # target_x_size = t_top_x - t_bottom_x
    # target_y_size = t_top_y - t_bottom_y
    # y_diff = target_y_size - source_y_size
    # x_diff = target_x_size - source_x_size

    target[t_bottom_y:t_top_y + 1, t_bottom_x:t_top_x + 1] = source[s_bottom_y:s_top_y + 1, s_bottom_x:s_top_x + 1]

def Get_Fundamental(img1, img2):
    descriptor = cv2.ORB_create(2000)
    kp1,des1 = descriptor.detectAndCompute(img1,None)
    kp2,des2 = descriptor.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck=True
    matches = bf.knnMatch(des1, des2, k=2)

    # matches = sorted(matches, key=lambda x: x.distance)
    pts1 =[]
    pts2 =[]
    for m,n in matches:
        if m.distance < 0.3 *n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)

    return F