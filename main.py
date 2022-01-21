import cv2
import glob
import numpy as np
from custom_vision_tool import *
import pickle
import os
from custom_metric import *
Lambda_1 = 0.15
Lambda_2 = 0.4
Lambda_3 = 0.45
sigma_c = 4.8
sigma_g = 0.25

butterfly_image = glob.glob("D:/Davis dataset/DAVIS/JPEGImages/480p/butterfly/*.jpg")
sampled_butterfly_image = [butterfly_image[i]  for i in range(0,len(butterfly_image), 10)]
saved_imgs = []
saved_imgs_gray = []
correspondence_map = []#denseflow에 대한 매핑입니다.
lab_images = []#lab feature에 대한 매핑입니다.
#load ref image and source image
Extracted_feature = [0 for _ in range(len(sampled_butterfly_image))]
fundamental_mats = []
for idx, image_name in enumerate(sampled_butterfly_image):
    sample_img = cv2.imread(image_name)
    saved_img = sample_img.copy()

    saved_imgs.append(saved_img)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    saved_imgs_gray.append(sample_img)
    Extracted_feature[idx] = {}
    if not os.path.isfile('denseSIFT.pickle'):
        print(idx, "번째 이미지를 읽는중입니다.")
        for i in range(saved_img.shape[0]):
            for j in range(saved_img.shape[1]):
                des = extract_SIFT(sample_img,j,i,6)
                if des is None:
                    exit(0)
                elif len(des) == 0:
                    Extracted_feature[idx][(i,j)] = 0
                else:
                    Extracted_feature[idx][(i,j)] =des

if not os.path.isfile('denseSIFT.pickle'):
    with open('denseSIFT.pickle', 'wb') as f:
        pickle.dump(Extracted_feature,f,pickle.HIGHEST_PROTOCOL)
else:
    with open('denseSIFT.pickle', 'rb') as f:
        Extracted_feature = pickle.load(f)


ref_img = saved_imgs_gray[0]

img_h, img_w = ref_img.shape
#max_value = 0 #기존에 노멀라이즈 해야하지만 이미 노멀라이즈 되어있습니다.
# for idx, image in enumerate(sampled_butterfly_image):
#     for i in range(ref_img.shape[0]):
#         for j in range(ref_img.shape[1]):
#             descriptor = Extracted_feature[idx][(i,j)]
#             if type(descriptor) is np.ndarray:
#                 norm_vec = np.linalg.norm(descriptor)
#                 print(descriptor)
#                 print(norm_vec)
#                 if max_value < norm_vec:
#                     max_value = norm_vec
source_imgs = saved_imgs_gray[1:]


similarity_map =np.zeros((img_h,img_w,len(source_imgs)))
for source_img in source_imgs:
    fundamental_matrix = Get_Fundamental(ref_img,source_img)
    fundamental_mats.append(fundamental_matrix)


for source_img in source_imgs:
    #get denseflow
    dense_flow = cv2.calcOpticalFlowFarneback(ref_img, source_img,None,0.5,3,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    correspondence_map.append(dense_flow)

for image in saved_imgs:
    #get lad feature
    lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    lab_images.append(lab_image)

for idx, source_img in enumerate(source_imgs):
    print(idx)
    for i in range(source_img.shape[0]):
        for j in range(source_img.shape[1]):
            #Extracted_feature의 값이 numpy인지를 확인합니다.
            # 아니라면 similarity_map을 채우지않고 넘어갑니다.
            if type(Extracted_feature[0][(i,j)]) is not np.ndarray:
                continue

            x_hat = np.array([i,j]) +np.flip(correspondence_map[idx][i,j])
            x_hat_int = x_hat.astype(np.int32)
            if(x_hat_int[0]<0 or x_hat_int[1]<0) or (x_hat_int[0] >= img_h or x_hat_int[1]>= img_w):
                continue#투영된 점
            if type(Extracted_feature[idx][(x_hat_int[0], x_hat_int[1])]) is not np.ndarray:
                similarity_map[i, j, idx] =-1
                continue
            f_c_x_r = lab_images[0][i,j,:]#ref이미지에 대한 LAB벡터입니다.
            f_c_x_hat = lab_images[idx][x_hat_int[0],x_hat_int[1],:]#source 이미지에 대한 LAB벡터입니다.

            f_g_x_r = Extracted_feature[0][(i,j)]#ref이미지에 대한 dense sift입니다.
            f_g_x_hat = Extracted_feature[idx][(x_hat_int[0],x_hat_int[1])]  # ref이미지에 대한 dense sift입니다.

            dist_1 =Se_distance(f_c_x_r,f_c_x_hat,sigma_c)
            dist_2 =Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
            dist_3 = Sf_distance(np.float64([i,j]),np.float64([x_hat[0],x_hat[1]]),fundamental_mats[idx])
            similarity_map[i, j, idx] = Lambda_1*dist_1+\
                                                Lambda_2*dist_2+\
                                                Lambda_3*dist_3
scan_neighbor = [[0,-1],[-1,0]],[[0,1],[1,0]]
for scan_vec in [0,1]:
    for i in range(saved_img.shape[0]):
        for j in range(saved_img.shape[1]):
            if type(Extracted_feature[0][(i,j)]) is not np.ndarray:
                continue
            x_hats = []
            for idx, source_img in enumerate(source_imgs):
                x_hats_in_source = []
                for neigh_point_delta in scan_neighbor[scan_vec]:
                    x_ref_neigh = np.float64([i, j])+np.float64(neigh_point_delta)
                    x_hat_neigh = x_ref_neigh + np.flip(correspondence_map[idx][int(x_ref_neigh[0]), int(x_ref_neigh[1])])
                    x_hat_cand = x_hat_neigh - np.float64(neigh_point_delta)
                    x_hat_cand = x_hat_cand.astype(np.int32)
                    x_hats_in_source.append(x_hat_cand)#후보점들을 추가합니다.
                x_hats.append(x_hats_in_source)

