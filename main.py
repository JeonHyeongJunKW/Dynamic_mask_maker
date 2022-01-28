import cv2
import glob
import numpy as np
from custom_vision_tool import *
import pickle5 as pickle
import os
from custom_metric import *
from sklearn.cluster import DBSCAN
from cyvlfeat.sift import dsift
import copy
Lambda_1 = 0.15
Lambda_2 = 0.4
Lambda_3 = 0
sigma_c = 4.8
sigma_g = 0.25
sigma_h = 4.8
Lambda_4 = Lambda_1/(Lambda_1+Lambda_2)
Lambda_5 = Lambda_2/(Lambda_1+Lambda_2)
Lambda_6 = 0.12
Lambda_7 = 0.36
Lambda_8 = 0.03
t_r =0.5

patchsize = 10 # 특징추출 및 이미지 갱신에 사용되는 패치사이즈입니다.

butterfly_image = glob.glob("D:/Davis dataset/DAVIS/JPEGImages/480p/butterfly/*.jpg")
sampled_butterfly_image = [butterfly_image[i]  for i in range(0,len(butterfly_image), 10)]#10개정도의 샘플이미지에서 장소이미지를 얻어옵니다.
saved_color_imgs = []
saved_imgs_gray = []
correspondence_map = []#denseflow에 대한 매핑입니다.
lab_images = [] # lab mean feature에 대한 매핑입니다.(소스이미지에 대한 매핑만이 존재합니다.)
#load ref image and source image
Extracted_feature = [0 for _ in range(len(sampled_butterfly_image))]#feature 추출은 모든 이미지(ref, source)에 대해서 합니다.
fundamental_mats = []#각각의 fundamental matrix입니다.

cast_feat = {}
color_bgr_img = cv2.imread(sampled_butterfly_image[0])
for y in range(color_bgr_img.shape[0]):
    for x in range(color_bgr_img.shape[1]):
        #특징들을 feature 개수만큼 만듭니다.
        cast_feat[(y, x)] = 0
max_height = 0
max_width = 0
min_height =color_bgr_img.shape[0]
min_width = color_bgr_img.shape[1]
for idx, image_name in enumerate(sampled_butterfly_image):
    color_bgr_img = cv2.imread(image_name)
    color_origin_img = color_bgr_img.copy()
    saved_color_imgs.append(color_origin_img)

    gray_img = cv2.cvtColor(color_bgr_img, cv2.COLOR_BGR2GRAY)
    saved_imgs_gray.append(gray_img)
    Extracted_feature[idx] = cast_feat.copy()
    float_img = gray_img.astype(np.float32)
    F = dsift(float_img, size=patchsize,float_descriptors=True)
    start = time.time()

    if not os.path.isfile('denseSIFT.pickle'):
        print(idx, "번째 이미지를 읽는중입니다.")
        keypoint = F[0].astype(np.int32)
        descriptor = F[1]
        for feat_idx in range(keypoint.shape[0]):
            des = np.reshape(descriptor[feat_idx, :], (1, -1))
            size = np.linalg.norm(des)
            if size !=0:
                des = des/size
            Extracted_feature[idx][(keypoint[feat_idx, 0], keypoint[feat_idx, 1])] = des
            if idx ==0:
                if keypoint[feat_idx, 0] >= max_height:
                    max_height = keypoint[feat_idx, 0]
                if keypoint[feat_idx, 1] >= max_width:
                    max_width = keypoint[feat_idx, 1]
                if keypoint[feat_idx, 0] <= min_height :
                    min_height = keypoint[feat_idx, 0]
                if keypoint[feat_idx, 1] <= min_width :
                    min_width = keypoint[feat_idx, 1]



ref_img = saved_imgs_gray[0]#기존 gray성격의 reference 이미지 입니다.
return_img = saved_color_imgs[0].copy()#수정되는 color형태의 이미지입니다.
img_h, img_w = ref_img.shape
mask_img_1 = np.ones((img_h,img_w), dtype=np.uint8)*255#이미지 마스킹간에 사용되는 마스크입니다.
mask_img_2 = np.ones((img_h,img_w), dtype=np.uint8)*255#이미지 마스킹간에 사용되는 마스크입니다.
source_imgs = saved_imgs_gray[1:]#첫번째 원소를 제외한 소스이미지입니다.


similarity_map =np.zeros((img_h,img_w,len(source_imgs)))#similarity map의 크기는 source 이미지 개수
for source_img in source_imgs:
    fundamental_matrix = Get_Fundamental(ref_img,source_img)
    fundamental_mats.append(fundamental_matrix)


for source_img in source_imgs:
    #get denseflow
    dense_flow = cv2.calcOpticalFlowFarneback(ref_img, source_img,None,0.5,3,13,10,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    correspondence_map.append(dense_flow)

for image in saved_color_imgs:
    #get lad feature
    lab_image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    kernel = np.ones((patchsize,patchsize),np.float32)/(patchsize*patchsize)
    lab_image = cv2.filter2D(lab_image,-1,kernel)
    lab_images.append(lab_image)

img_canny = cv2.Canny(image,50, 200)
have_edge_test= np.zeros((source_img.shape[0],source_img.shape[1]))
for y in range(source_img.shape[0]):
    for x in range(source_img.shape[1]):
        bottom_x = int(x - patchsize / 2) if int(x - patchsize / 2) >= 0 else 0
        bottom_y = int(y - patchsize / 2) if int(y - patchsize / 2) >= 0 else 0
        top_x = int(x + patchsize / 2) if int(x + patchsize / 2) < img_canny.shape[1] else img_canny.shape[1] - 1
        top_y = int(y + patchsize / 2) if int(y + patchsize / 2) < img_canny.shape[0] else img_canny.shape[0] - 1
        test_patch = img_canny[bottom_y:top_y,bottom_x:top_x]
        edge_have = np.where(test_patch==255)
        if edge_have[0].shape[0] >0:
            have_edge_test[y,x] = 255
# cv2.imshow("test",have_edge_test)
# cv2.waitKey(0)
for idx, source_img in enumerate(source_imgs):#일종에 신뢰도를 구하기 위한 전처리과정인데 너무 오래걸린다.
    print("making similarity map :",idx)
    for y in range(source_img.shape[0]):
        for x in range(source_img.shape[1]):
            #Extracted_feature의 값이 numpy인지를 확인합니다.
            # 아니라면 similarity_map을 채우지않고 넘어갑니다.
            if type(Extracted_feature[0][(y, x)]) is not np.ndarray:
                continue

            x_hat = np.array([y, x]) + np.flip(correspondence_map[idx][y, x])
            x_hat_int = x_hat.astype(np.int32)
            if(x_hat_int[0]<0 or x_hat_int[1]<0) or (x_hat_int[0] >= img_h or x_hat_int[1]>= img_w):
                continue#이미지 밖으로 투영된 점
            if type(Extracted_feature[idx+1][(x_hat_int[0], x_hat_int[1])]) is not np.ndarray:
                similarity_map[y, x, idx] =0
                continue

            f_g_x_r = Extracted_feature[0][(y, x)]#ref이미지에 대한 dense sift입니다.
            f_g_x_hat = Extracted_feature[idx+1][(x_hat_int[0],x_hat_int[1])]  # ref이미지에 대한 dense sift입니다.

            dist_2 =Se_distance(f_g_x_r, f_g_x_hat, 0.35)
            similarity_map[y, x, idx] = dist_2

#탐생방향에 대한 파라미터
scan_neighbor = [[0,-1],[-1,0]],[[0,1],[1,0]]
start_height = [min_height, max_height]
start_width = [min_width, max_width]
end_height = [max_height+1, min_height-1]
end_width = [max_width+1, min_width-1]
way = [1,-1]
# print("main start")
initial_correspondence_map = correspondence_map.copy()
# initial_Extracted_feature = copy.deepcopy(Extracted_feature)
initial_similarity_map= similarity_map.copy()
for scan_vec in [1,0]:
    if scan_vec == 1:
        mask = mask_img_1
        origin_color = [221,0,255]
    else :
        mask = mask_img_2
        origin_color = [90, 243, 197]
        correspondence_map = initial_correspondence_map
        similarity_map= initial_similarity_map

    for y in range(start_height[scan_vec], end_height[scan_vec], way[scan_vec]):
        for x in range(start_width[scan_vec], end_width[scan_vec], way[scan_vec]):
            # cv2.imshow("mask", mask_img)
            cv2.imshow("result", return_img)
            cv2.waitKey(1)
            if type(Extracted_feature[0][(y, x)]) is not np.ndarray: #만약에 레퍼런스 이미지가 feature가 존재하지않는다면
                continue

            x_hats = []
            for idx, source_img in enumerate(source_imgs):
                x_hats_in_source = []
                #각 소스 이미지마자 후보점들을 추가합니다.
                for neigh_point_delta in scan_neighbor[scan_vec]:
                    x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                    x_hat_neigh = x_ref_neigh + np.flip(correspondence_map[idx][int(x_ref_neigh[0]), int(x_ref_neigh[1])])#동적인 물체 주변에서는 완전히 이상한 값이 나옴
                    x_hat_cand = x_hat_neigh - np.float64(neigh_point_delta)
                    x_hat_cand = x_hat_cand.astype(np.int32)
                    x_hats_in_source.append(x_hat_cand)#후보점들을 추가합니다.

                x_hats.append(x_hats_in_source)
            # print(x_hats)
            ##change candidate points to features
            x_hats_denseSIFT =0
            x_hats_confidence = []  # 각 후보점에 대한 신뢰도를 조사합니다.
            x_hats_lab = 0
            x_hats_c_x2_neigh_idx = [] # 후보점 선택에 참여한 인덱스를 얻습니다.
            good_x_hat = []# 후보선택에 사용된 x_hat
            ind =0
            Extracted_feature_Size =0
            for idx, one_source_candidate in enumerate(x_hats):
                for point in one_source_candidate:
                    ind +=1
                    if (point[0] <0 or point[0] >= img_h) or (point[1] <0 or point[1] >= img_w):
                        continue
                    if type(Extracted_feature[idx+1][(point[0],point[1])]) is not np.ndarray:
                        continue#If 0, this is the point where we didn't get the feature.
                    if type(x_hats_denseSIFT) is not np.ndarray:
                        x_hats_denseSIFT = Extracted_feature[idx + 1][(point[0], point[1])]
                        Extracted_feature_Size = x_hats_denseSIFT.shape[1]#각 feature의 size입니다.
                        DBSCAN_features = x_hats_denseSIFT
                    else:
                        x_hats_denseSIFT = Extracted_feature[idx + 1][(point[0], point[1])]
                        DBSCAN_feature = x_hats_denseSIFT
                        DBSCAN_features = np.vstack((DBSCAN_features,DBSCAN_feature))
                    x_hats_confidence.append(similarity_map[point[0], point[1], idx])
                    x_hats_c_x2_neigh_idx.append(ind - 1)
                    good_x_hat.append([point[0], point[1]])
            if type(x_hats_denseSIFT) is not np.ndarray:
                continue#후보 feature들이 모두 자격이 없는경우
            dbscan_model = DBSCAN(eps=0.1,min_samples=1)
            clustering = dbscan_model.fit(DBSCAN_features)
            clustering_label = clustering.labels_
            # print(clustering_label)
            k= np.max(clustering_label)#군집의 갯수 최소 샘플은 1로했기때문에 outlier취급은 하지않는다.
            max_cluster_idx =-1
            max_b_k = 0
            feature_conf = np.array(x_hats_confidence)#각 신뢰도를 numpy로 변형합니다.
            # print("이게 현재 feature conf다 ",feature_conf)
            cluster_by_bk = {}
            for cluster_idx in range(k+1):#0부터 가장 큰 라벨에 대해서 검사를 합니다.
                k_full_feature = DBSCAN_features[clustering_label == cluster_idx, :]#해당 클러스터의 feature를 가져옵니다.
                k_feature_conf = feature_conf[clustering_label == cluster_idx]
                # print(k_feature_conf," : ",np.sum(k_feature_conf)," - ",k_full_feature.shape)
                b_k = np.sum(k_feature_conf)
                cluster_by_bk[cluster_idx] = b_k
                if b_k> max_b_k:
                    max_b_k = b_k
                    max_cluster_idx = cluster_idx
            if max_cluster_idx ==-1 :#단일 클러스터이며, 가장자리에 해당하는부분이라서 dynamic object를 잡기힘들다.
                continue
            #가장 static한 지점이라는 점에 대해서 연산합니다.
            k_full_feature = DBSCAN_features[clustering_label == max_cluster_idx, :]
            # print(max_cluster_idx)
            # print(cluster_by_bk)
            k_feature_conf = feature_conf[clustering_label == max_cluster_idx]
            center_of_denseSIFT = k_full_feature[0, :] * k_feature_conf[0]

            for idx_k_max in range(1,k_full_feature.shape[0]):
                center_of_denseSIFT += k_full_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
            center_of_denseSIFT /= max_b_k


            sum_score = 0.0
            full_Z = 0.0

            for idx, source_img in enumerate(source_imgs):
                #각 소스 이미지마자 후보점들을 추가합니다.
                cand_point = np.float64([y, x])+ np.flip(correspondence_map[idx][y, x])
                if (cand_point[0] < 0 or cand_point[0] >= img_h) or (cand_point[1] < 0 or cand_point[1] >= img_w):
                    continue
                d = Sf_distance2(np.float64([x,y,1]),np.float64([cand_point[1],cand_point[0],1]),fundamental_mats[idx])
                conf_s = similarity_map[y,x,idx]
                full_Z +=conf_s
                sum_score +=conf_s*d
            if full_Z != 0:
                sum_score = sum_score / full_Z

            new_sigma_3 =20
            final_s_x = np.exp(-sum_score*sum_score/(2*new_sigma_3*new_sigma_3))
            #물체의 경계주변에서는 경쟁자 패치가 잘 안먹힘. 그래서 fundamental이 잘먹힘
            M_xr = 0
            # if have_edge_test[y,x] ==255: #해당패치가 엣지를 가지고 있다면?
            if False:
                M_xr = (1-0.5)*Se_distance(Extracted_feature[0][(y, x)], center_of_denseSIFT, 0.45) + \
                       0.5*final_s_x
            else : #엣지가 없다면 굳이 기하학 따질 필요없음
                M_xr = Se_distance(Extracted_feature[0][(y, x)], center_of_denseSIFT, 0.45)

            if M_xr <= t_r: # 임계값보다 작으면 동적인 물체!
                if mask_img_1[y, x] == 255:
                    return_img[y, x] = origin_color#동적인 물체로 마스킹합니다.
                    mask_img_1[y, x] = 244
                else :
                    return_img[y, x] = [0,0,255]
                    mask_img_1[y, x] =0
                # print(M_xr)
                test_idx = 0
                cluster_idx = 0
                for source_ind in range(len(source_imgs)):
                    max_cost = 0
                    max_source_cand_i = -1
                    for source_cand_i in range(2):
                        if not test_idx in x_hats_c_x2_neigh_idx:
                            test_idx +=1
                            continue
                        current_cand_label = clustering_label[cluster_idx]
                        cost = cluster_by_bk[current_cand_label]
                        if cost >max_cost:
                            max_source_cand_i = source_cand_i
                            max_cost = cost
                        test_idx +=1
                    if max_source_cand_i ==-1:
                        continue
                    else :
                        # 실제 후보점을 좌표로 넣고
                        # print(x_hats_in_source)
                        # print(max_source_cand_i)
                        new_point = np.flip(x_hats[source_ind][max_source_cand_i] - np.float64([y,x]))
                        correspondence_map[source_ind][y,x] = new_point
                        neigh_point_delta = scan_neighbor[scan_vec][max_source_cand_i]
                        x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                        # 신뢰도는 이웃했던 정적인점의 신뢰도를 대신해서 넣는다.
                        similarity_map[y, x, source_ind] = similarity_map[int(x_ref_neigh[0]),int(x_ref_neigh[1]),source_ind]


cv2.imshow("mask", mask_img_1)
cv2.waitKey(0)