import cv2
import glob
import numpy as np
from feature_extract import extract_SIFT
butterfly_image = glob.glob("D:/Davis dataset/DAVIS/JPEGImages/480p/butterfly/*.jpg")
sampled_butterfly_image = [butterfly_image[i]  for i in range(0,len(butterfly_image), 10)]
saved_imgs = []
saved_imgs_gray = []
#load ref image and source image
Extracted_feature = [0 for _ in range(len(sampled_butterfly_image))]
for idx, image_name in enumerate(sampled_butterfly_image):
    sample_img = cv2.imread(image_name)
    saved_img = sample_img.copy()

    saved_imgs.append(saved_img)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    saved_imgs_gray.append(sample_img)
    Extracted_feature[idx] = {}
    print("나 이제 시작")
    for i in range(saved_img.shape[0]):
        for j in range(saved_img.shape[1]):
            des = extract_SIFT(sample_img,j,i,6)
            if des is None:
                exit(0)
            elif len(des) == 0:
                Extracted_feature[idx][(i,j)] = 0
            else:
                Extracted_feature[idx][(i,j)] =des
    print(idx, " : 종료")

ref_img = saved_imgs_gray[0]
img_h, img_w = ref_img.shape

source_imgs = saved_imgs_gray[1:]

correspondence_map = []

for source_img in source_imgs:
    #get denseflow
    dense_flow = cv2.calcOpticalFlowFarneback(ref_img, source_img,None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    correspondence_map.append(dense_flow)

