import cv2
import glob
import numpy as np
from custom_vision_tool import *
import pickle
import os
from custom_metric import *
from sklearn.cluster import DBSCAN
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
t_r =0.4

patchsize = 32 # 특징추출 및 이미지 갱신에 사용되는 패치사이즈입니다.

butterfly_image = glob.glob("D:/Davis dataset/DAVIS/JPEGImages/480p/butterfly/*.jpg")
sampled_butterfly_image = [butterfly_image[i]  for i in range(0,len(butterfly_image), 10)]#10개정도의 샘플이미지에서 장소이미지를 얻어옵니다.
saved_color_imgs = []
saved_imgs_gray = []
correspondence_map = []#denseflow에 대한 매핑입니다.
lab_images = [] # lab mean feature에 대한 매핑입니다.(소스이미지에 대한 매핑만이 존재합니다.)
#load ref image and source image
Extracted_feature = [0 for _ in range(len(sampled_butterfly_image))]#feature 추출은 모든 이미지(ref, source)에 대해서 합니다.
fundamental_mats = []#각각의 fundamental matrix입니다.
for idx, image_name in enumerate(sampled_butterfly_image):
    color_bgr_img = cv2.imread(image_name)
    color_origin_img = color_bgr_img.copy()
    saved_color_imgs.append(color_origin_img)

    gray_img = cv2.cvtColor(color_bgr_img, cv2.COLOR_BGR2GRAY)
    saved_imgs_gray.append(gray_img)
    Extracted_feature[idx] = {}
    if not os.path.isfile('denseSIFT.pickle'):
        print(idx, "번째 이미지를 읽는중입니다.")
        for y in range(color_bgr_img.shape[0]):
            for x in range(color_bgr_img.shape[1]):
                des = extract_SIFT(gray_img, x, y, patchsize)#패치사이즈 만하게 descriptor를 뽑습니다.
                if len(des) == 0:#가장자리 영역이면 해당 descriptor를 0으로 초기화합니다.
                    Extracted_feature[idx][(y, x)] = 0
                else:#아니라면 해당 descriptor를 descriptor로 초기화합니다. 1 x120이엇나..
                    Extracted_feature[idx][(y, x)] =des

if not os.path.isfile('denseSIFT.pickle'):
    with open('denseSIFT.pickle', 'wb') as f:
        pickle.dump(Extracted_feature,f,pickle.HIGHEST_PROTOCOL)
else:
    with open('denseSIFT.pickle', 'rb') as f:
        Extracted_feature = pickle.load(f)

ref_img = saved_imgs_gray[0]#기존 gray성격의 reference 이미지 입니다.
return_img = saved_color_imgs[0].copy()#수정되는 color형태의 이미지입니다.
img_h, img_w = ref_img.shape
mask_img = np.ones((img_h,img_w))*255#이미지 마스킹간에 사용되는 마스크입니다.
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



if not os.path.isfile('similarity_map.pickle'):
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
                f_c_x_r = lab_images[0][y, x, :]#ref이미지에 대한 LAB벡터입니다.
                f_c_x_hat = lab_images[idx+1][x_hat_int[0],x_hat_int[1],:]#source 이미지에 대한 LAB벡터입니다.

                f_g_x_r = Extracted_feature[0][(y, x)]#ref이미지에 대한 dense sift입니다.
                f_g_x_hat = Extracted_feature[idx+1][(x_hat_int[0],x_hat_int[1])]  # ref이미지에 대한 dense sift입니다.

                dist_1 =Se_distance(f_c_x_r,f_c_x_hat,sigma_c)
                dist_2 =Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                dist_3 = Sf_distance(np.float64([x, y, 1]), np.float64([x_hat[1], x_hat[0], 1]), fundamental_mats[idx])
                similarity_map[y, x, idx] = Lambda_1 * dist_1 + \
                                            Lambda_2 * dist_2 + \
                                            Lambda_3 * dist_3
if not os.path.isfile('similarity_map.pickle'):
    with open('similarity_map.pickle', 'wb') as f:
        pickle.dump(similarity_map,f,pickle.HIGHEST_PROTOCOL)
else:
    with open('similarity_map.pickle', 'rb') as f:
        similarity_map = pickle.load(f)

#탐생방향에 대한 파라미터
scan_neighbor = [[0,-1],[-1,0]],[[0,1],[1,0]]
start_height = [0,ref_img.shape[0]-1]
start_width = [0, ref_img.shape[1] - 1]
end_height = [ref_img.shape[0]-1,-1]
end_width = [ref_img.shape[1]- 1, -1]
way = [1,-1]
print("main start")
for scan_vec in [1,0]:
    for y in range(start_height[scan_vec], end_height[scan_vec], way[scan_vec]):
        for x in range(start_width[scan_vec], end_width[scan_vec], way[scan_vec]):
            if type(Extracted_feature[0][(y, x)]) is not np.ndarray: #만약에 레퍼런스 이미지가 feature가 존재하지않는다면
                continue
            x_hats = []
            # print(y,x)
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
            DBSCAN_features = 0#DBSCAN간에 사용할 metric함수를 위한 feature 집합
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
                        x_hats_lab = np.reshape(lab_images[idx+1][point[0], point[1],:],(1,3))
                        DBSCAN_features = np.reshape(np.concatenate((x_hats_denseSIFT[0], np.reshape(x_hats_lab,(1,3))[0])),(1,-1))
                    else:
                        x_hats_denseSIFT= np.vstack((x_hats_denseSIFT, Extracted_feature[idx + 1][(point[0], point[1])]))
                        x_hats_lab = np.vstack((x_hats_lab,lab_images[idx+1][point[0], point[1], :]))
                        DBSCAN_feature = np.concatenate((Extracted_feature[idx + 1][(point[0], point[1])][0], np.reshape(lab_images[idx+1][point[0], point[1], :],(1,3))[0]))
                        DBSCAN_features = np.vstack((DBSCAN_features,DBSCAN_feature))
                    x_hats_confidence.append(similarity_map[point[0], point[1], idx])
                    x_hats_c_x2_neigh_idx.append(ind - 1)
                    good_x_hat.append([point[0], point[1]])
            if type(x_hats_denseSIFT) is not np.ndarray:
                continue#후보 feature들이 모두 자격이 없는경우
            # print(Extracted_feature_Size)
            dbscan_model = DBSCAN(eps=0.1,min_samples=1,metric=clustering_feature)
            clustering = dbscan_model.fit(DBSCAN_features)
            clustering_label = clustering.labels_
            k= np.max(clustering_label)#군집의 갯수 최소 샘플은 1로했기때문에 outlier취급은 하지않는다.
            max_cluster_idx =-1
            max_b_k = 0
            feature_conf = np.array(x_hats_confidence)#각 신뢰도를 numpy로 변형합니다.
            # print("이게 현재 feature conf다 ",feature_conf)
            for cluster_idx in range(k+1):#0부터 가장 큰 라벨에 대해서 검사를 합니다.
                k_full_feature = x_hats_denseSIFT[clustering_label == cluster_idx, :]#해당 클러스터의 feature를 가져옵니다.
                k_feature_conf =feature_conf[clustering_label ==cluster_idx]
                # print(k_feature_conf," : ",np.sum(k_feature_conf)," - ",k_full_feature.shape)
                b_k = np.sum(k_feature_conf)
                if b_k> max_b_k:
                    max_b_k = b_k
                    max_cluster_idx = cluster_idx
            if max_cluster_idx ==-1 :#단일 클러스터이며, 가장자리에 해당하는부분이라서 dynamic object를 잡기힘들다.
                continue
            #가장 static한 지점이라는 점에 대해서 연산합니다.
            k_full_feature = x_hats_denseSIFT[clustering_label == max_cluster_idx, :]
            # print(x_hats_lab)
            k_LAB_feature = x_hats_lab[clustering_label == max_cluster_idx]
            k_feature_conf = feature_conf[clustering_label == max_cluster_idx]
            center_of_denseSIFT = k_full_feature[0, :] * k_feature_conf[0]
            center_of_LAB = k_LAB_feature[0, :] * k_feature_conf[0]

            for idx_k_max in range(1,k_full_feature.shape[0]):
                center_of_denseSIFT += k_full_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
                center_of_LAB += k_LAB_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
            center_of_denseSIFT /= max_b_k
            center_of_LAB /= max_b_k
            M_xr = Lambda_4*Se_distance(lab_images[0][y, x, :], center_of_LAB, sigma_c)\
                    + Lambda_5*Se_distance(Extracted_feature[0][(y, x)], center_of_denseSIFT, sigma_g)

            if M_xr < t_r: # 임계값보다 작으면 동적인 물체!
                # 다른 프레임에서의 레퍼런스점 x^a_s가 후보점일 수도 있다.
                # Q_x^a_s는  source image 에서 얻은 패치이다.
                # P는 그중에서 Am에 속하는 점들의 패치 집합이다.
                # W^1_r은 q에 대한 이미지 패치이다.
                mask_img[y, x] =0#동적인 물체로 마스킹합니다.
                Am_origin = np.array(x_hats_c_x2_neigh_idx)[clustering_label == max_cluster_idx]#가장 신뢰도가 높은 k의 점들
                print("각 후보점들의 기본인덱스", Am_origin)
                Am_good_x_hat = np.array(good_x_hat)[clustering_label == max_cluster_idx,:]
                Am = (Am_origin/2).astype(np.int32).tolist()#각각의 source 이미지 단위로 표시
                print("각 후보점들의 소스채널", Am+1)
                max_q = -1
                q_value = 0
                for id_q, source_ind in enumerate(Am):
                    first_point = x_hats[source_ind][0]
                    second_point = x_hats[source_ind][1]
                    if (first_point[0] < 0 or first_point[0] >= img_h) or (first_point[1] < 0 or first_point[1] >= img_w):
                        continue
                    if (second_point[0] < 0 or second_point[0] >= img_h) or (second_point[1] < 0 or second_point[1] >= img_w):
                        continue
                    first_point_DSIFT = Extracted_feature[source_ind + 1][(first_point[0], first_point[1])]

                    second_point_DSIFT = Extracted_feature[source_ind + 1][(second_point[0], second_point[1])]
                    ref_DSIFT = Extracted_feature[0][(y, x)]

                    first_Se_c = Se_distance(lab_images[source_ind+1][first_point[0],first_point[1],:],
                                             lab_images[0][y, x, :], sigma_c)
                    first_Se_h = Se_distance(first_point_DSIFT, ref_DSIFT, sigma_h)

                    second_Se_c = Se_distance(lab_images[source_ind+1][second_point[0], second_point[1], :],
                                              lab_images[0][y, x, :], sigma_c)
                    second_Se_h = Se_distance(second_point_DSIFT, ref_DSIFT, sigma_h)
                    # 이부분이 Am외적인 부분임
                    q_temp = Lambda_6*first_Se_c + \
                             Lambda_6*second_Se_c + \
                             Lambda_7* first_Se_h + \
                             Lambda_7* second_Se_h + \
                             Lambda_8*Sf_distance(np.float64([x, y, 1])\
                                                  ,np.float64([Am_good_x_hat[id_q][1],Am_good_x_hat[id_q][0],1])\
                                                  ,fundamental_mats[source_ind])
                    if q_value<q_temp:
                        q_value = q_temp
                        max_q = id_q
                #update I, fc , Fg
                final_cand_point = Am_good_x_hat[max_q]
                final_fc = lab_images[Am[max_q]+1][final_cand_point[0],final_cand_point[1],:]
                final_fg = Extracted_feature[Am[max_q]+1][(final_cand_point[0], final_cand_point[1])]
                #패치를 대체합니다.
                return_img[y, x] = saved_color_imgs[Am[max_q] + 1][final_cand_point[0], final_cand_point[1], :]
                change_patch(saved_color_imgs[Am[max_q] + 1], final_cand_point[0], final_cand_point[1], return_img, y, x, 4)
                lab_images[0][y, x] = final_fc
                Extracted_feature[0][(y, x)] = final_fg

                #feature mapping과 c를 수정합니다.
                Am_lists = np.array(list(Am))
                list_q_in_AM = np.where(Am_lists ==Am[max_q])
                if list_q_in_AM[0].shape[0] ==1:

                    correspondence_map[Am[max_q]][y, x] = np.flip(final_cand_point - np.array([y, x]))#현재 다른차원에서 매핑된 점을 사용합니다.
                    f_c_x_r = lab_images[0][y, x, :]  # ref이미지에 대한 LAB벡터입니다.
                    f_c_x_hat = lab_images[Am[max_q] + 1][final_cand_point[0], final_cand_point[1], :]  # source 이미지에 대한 LAB벡터입니다.

                    f_g_x_r = Extracted_feature[0][(y, x)]  # ref이미지에 대한 dense sift입니다.
                    f_g_x_hat = Extracted_feature[Am[max_q]+1][(final_cand_point[0], final_cand_point[1])]  # ref이미지에 대한 dense sift입니다.

                    dist_1 = Se_distance(f_c_x_r, f_c_x_hat, sigma_c)
                    dist_2 = Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                    dist_3 = Sf_distance(np.float64([x, y,1]), np.float64([final_cand_point[1], final_cand_point[0],1]), fundamental_mats[Am[max_q]])
                    if np.isnan(dist_3):
                        print([y, x], [final_cand_point[0], final_cand_point[1]])
                        print("Am에 속하는 점이 1개일 경우, 패치사이에 너무 큰 fundamental 에러가 나왓습니다. 잘못된 매칭인듯합니다.")
                        cv2.waitKey(0)
                    similarity_map[y, x, Am[max_q]] = Lambda_1 * dist_1 + \
                                                      Lambda_2 * dist_2 + \
                                                      Lambda_3 * dist_3
                elif list_q_in_AM[0].shape[0] ==2:
                    best_cand_points = [Am_good_x_hat[list_q_in_AM[0][0]],Am_good_x_hat[list_q_in_AM[0][1]]]
                    best_sim =0
                    best_k = -1
                    for best_cand_k, best_cand_point in enumerate(best_cand_points):
                        f_c_x_r = lab_images[0][y, x, :]  # ref이미지에 대한 LAB벡터입니다.
                        f_c_x_hat = lab_images[Am[max_q] + 1][best_cand_point[0], best_cand_point[1], :]  # source 이미지에 대한 LAB벡터입니다.

                        f_g_x_r = Extracted_feature[0][(y, x)]  # ref이미지에 대한 dense sift입니다.
                        f_g_x_hat = Extracted_feature[Am[max_q]+1][(best_cand_point[0], best_cand_point[1])]  # ref이미지에 대한 dense sift입니다.

                        dist_1 = Se_distance(f_c_x_r, f_c_x_hat, sigma_c)
                        dist_2 = Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                        dist_3 = Sf_distance(np.float64([x, y,1]), np.float64([best_cand_point[1], best_cand_point[0],1]), fundamental_mats[Am[max_q]])
                        if np.isnan(dist_3):
                            print([y, x], [best_cand_point[0], best_cand_point[1]])
                            print("Am에 속하는 점이 2개일 경우, 패치사이에 너무큰 fundamental 에러가 나왓습니다. 잘못된 매칭인듯합니다.")
                            cv2.waitKey(0)
                        test_best_sim = Lambda_1 * dist_1 + \
                                                    Lambda_2 * dist_2 + \
                                                    Lambda_3 * dist_3
                        if test_best_sim >best_sim:
                            best_sim = test_best_sim
                            best_k =best_cand_k
                    if best_k ==-1:
                        print(best_cand_points)
                        print(test_best_sim)
                        print(best_sim)
                        print("이게 왜이럼ㅋㅋ")
                    similarity_map[y, x, Am[max_q]] = best_sim
                    correspondence_map[Am[max_q]][y, x] = np.flip(best_cand_points[best_k] - np.array([y, x]))  # 현재 다른차원에서 매핑된 점을 사용합니다.
                else:
                    print("z")
                cv2.imshow("mask", mask_img)
                cv2.imshow("result", return_img)
                cv2.waitKey(1)
cv2.waitKey(0)