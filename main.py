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
Lambda_3 = 0.45
sigma_c = 4.8
sigma_g = 0.25
sigma_h = 4.8
Lambda_4 = Lambda_1/(Lambda_1+Lambda_2)
Lambda_5 = Lambda_2/(Lambda_1+Lambda_2)
Lambda_6 = 0.12
Lambda_7 = 0.36
Lambda_8 = 0.03
t_r =0.1

patchsize = 6
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
                des = extract_SIFT(sample_img,j,i,patchsize)
                if len(des) == 0:
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
return_img = saved_imgs[0].copy()
img_h, img_w = ref_img.shape
mask_img = np.ones((img_h,img_w))*255
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
    kernel = np.ones((patchsize,patchsize),np.float32)/(patchsize*patchsize)
    lab_image = cv2.filter2D(lab_image,-1,kernel)
    lab_images.append(lab_image)



if not os.path.isfile('similarity_map.pickle'):
    for idx, source_img in enumerate(source_imgs):#일종에 신뢰도를 구하기 위한 전처리과정인데 너무 오래걸린다.
        print("making similarity map :",idx)
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
                    similarity_map[i, j, idx] =0
                    continue
                f_c_x_r = lab_images[0][i,j,:]#ref이미지에 대한 LAB벡터입니다.
                f_c_x_hat = lab_images[idx+1][x_hat_int[0],x_hat_int[1],:]#source 이미지에 대한 LAB벡터입니다.

                f_g_x_r = Extracted_feature[0][(i,j)]#ref이미지에 대한 dense sift입니다.
                f_g_x_hat = Extracted_feature[idx][(x_hat_int[0],x_hat_int[1])]  # ref이미지에 대한 dense sift입니다.

                dist_1 =Se_distance(f_c_x_r,f_c_x_hat,sigma_c)
                dist_2 =Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                dist_3 = Sf_distance(np.float64([i,j]),np.float64([x_hat[0],x_hat[1]]),fundamental_mats[idx])
                similarity_map[i, j, idx] = Lambda_1*dist_1+\
                                                    Lambda_2*dist_2+\
                                                    Lambda_3*dist_3
if not os.path.isfile('similarity_map.pickle'):
    with open('similarity_map.pickle', 'wb') as f:
        pickle.dump(similarity_map,f,pickle.HIGHEST_PROTOCOL)
else:
    with open('similarity_map.pickle', 'rb') as f:
        similarity_map = pickle.load(f)

#탐생방향에 대한 파라미터
scan_neighbor = [[0,-1],[-1,0]],[[0,1],[1,0]]
start_height = [0,saved_img.shape[0]-1]
start_width = [0, saved_img.shape[1] - 1]
end_height = [saved_img.shape[0]-1,-1]
end_width = [saved_img.shape[1]- 1, -1]
way = [1,-1]
print("main start")
for scan_vec in [1,0]:
    for i in range(start_height[scan_vec],end_height[scan_vec],way[scan_vec]):
        for j in range(start_width[scan_vec],end_width[scan_vec],way[scan_vec]):
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

            ##change candidate points to features
            full_feature =0
            x_hats_conf = []  # 각 후보점에 대한 신뢰도를 조사합니다.
            x_hats_lab = 0
            full_feature_list = [] # 후보점 선택에 참여한 인덱스를 얻습니다.
            good_x_hat = []# 후보선택에 사용된 x_hat
            ind =0
            for idx, one_source_candidate in enumerate(x_hats):
                for point in one_source_candidate:
                    ind +=1
                    if (point[0] <0 or point[0] >= img_h) or (point[1] <0 or point[1] >= img_w):
                        continue
                    if type(Extracted_feature[idx][(point[0],point[1])]) is not np.ndarray:
                        continue#If 0, this is the point where we didn't get the feature.
                    if type(full_feature) is not np.ndarray:
                        full_feature = Extracted_feature[idx][(point[0], point[1])]
                        x_hats_lab = lab_images[idx+1][point[0], point[1],:]
                    else:
                        full_feature= np.vstack((full_feature,Extracted_feature[idx][(point[0], point[1])]))
                        x_hats_lab = np.vstack((x_hats_lab,lab_images[idx+1][point[0], point[1], :]))
                    x_hats_conf.append(similarity_map[point[0], point[1], idx])
                    full_feature_list.append(ind-1)
                    good_x_hat.append([point[0], point[1]])
            if type(full_feature) is not np.ndarray:
                continue#후보 feature들이 모두 자격이 없는경우

            dbscan_model = DBSCAN(eps=0.1,min_samples=1)
            clustering = dbscan_model.fit(full_feature)
            clustering_label = clustering.labels_
            k= np.max(clustering_label)#군집의 갯수 최소 샘플은 1로했기때문에 outlier취급은 하지않는다.
            max_cluster_idx =-1
            max_b_k = 0
            feature_conf = np.array(x_hats_conf)#각 신뢰도를 numpy로 변형합니다.

            for cluster_idx in range(k):
                k_full_feature =full_feature[clustering_label ==cluster_idx,:]#해당 클러스터의 feature를 가져옵니다.
                k_feature_conf =feature_conf[clustering_label ==cluster_idx]
                # print(k_feature_conf," : ",np.sum(k_feature_conf)," - ",k_full_feature.shape)
                b_k = np.sum(k_feature_conf)
                if b_k> max_b_k:
                    max_b_k = b_k
                    max_cluster_idx = cluster_idx
            if max_cluster_idx ==-1 :#단일 클러스터이며, 가장자리에 해당하는부분이라서 dynamic object를 잡기힘들다.
                continue
            #가장 static한 지점이라는 점에 대해서 연산합니다.
            k_full_feature = full_feature[clustering_label == max_cluster_idx, :]
            k_LAB_feature = x_hats_lab[clustering_label == max_cluster_idx]
            k_feature_conf = feature_conf[clustering_label == max_cluster_idx]
            center_of_denseSIFT = k_full_feature[0, :] * k_feature_conf[0]
            center_of_LAB = k_LAB_feature[0, :] * k_feature_conf[0]

            for idx_k_max in range(1,k_full_feature.shape[0]):
                center_of_denseSIFT += k_full_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
                center_of_LAB += k_LAB_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
            center_of_denseSIFT /=max_b_k
            center_of_LAB /= max_b_k
            M_xr = Lambda_4*Se_distance(lab_images[0][i,j,:],center_of_LAB,sigma_c)\
                    +Lambda_5*Se_distance(Extracted_feature[0][(i,j)],center_of_denseSIFT,sigma_g)

            if M_xr < t_r: # 임계값보다 작으면 동적인 물체!
                # 다른 프레임에서의 레퍼런스점 x^a_s가 후보점일 수도 있다.
                # Q_x^a_s는  source image 에서 얻은 패치이다.
                # P는 그중에서 Am에 속하는 점들의 패치 집합이다.
                # W^1_r은 q에 대한 이미지 패치이다.
                mask_img[i,j] =0
                Am_origin = np.array(full_feature_list)[clustering_label == max_cluster_idx]#가장 신뢰도가 높은 k의 점들
                Am_good_x_hat = np.array(good_x_hat)[clustering_label == max_cluster_idx,:]
                Am = (Am_origin/2).astype(np.int32).tolist()#각각의 source 이미지 단위로 표시
                max_q = -1
                q_value = 0
                for id_q, source_ind in enumerate(Am):
                    first_point = x_hats[source_ind][0]
                    second_point = x_hats[source_ind][1]
                    if (first_point[0] < 0 or first_point[0] >= img_h) or (first_point[1] < 0 or first_point[1] >= img_w):
                        continue
                    if (second_point[0] < 0 or second_point[0] >= img_h) or (second_point[1] < 0 or second_point[1] >= img_w):
                        continue
                    first_point_Hog = Extracted_feature[idx][(first_point[0],first_point[1])]

                    second_point_Hog = Extracted_feature[idx][(second_point[0],second_point[1])]
                    ref_Hog = Extracted_feature[0][(i,j)]

                    first_Se_c = Se_distance(lab_images[source_ind+1][first_point[0],first_point[1],:],
                                           lab_images[0][i,j,:],sigma_c)
                    first_Se_h = Se_distance(first_point_Hog,ref_Hog,sigma_h)

                    second_Se_c = Se_distance(lab_images[source_ind+1][second_point[0], second_point[1], :],
                                           lab_images[0][i, j,:], sigma_c)
                    second_Se_h = Se_distance(second_point_Hog, ref_Hog, sigma_h)
                    # 이부분이 Am외적인 부분임
                    q_temp = Lambda_6*first_Se_c + \
                             Lambda_6*second_Se_c + \
                             Lambda_7* first_Se_h + \
                             Lambda_7* second_Se_h + \
                             Lambda_8*Sf_distance(np.float64([i,j])\
                                                  ,Am_good_x_hat[id_q].astype(np.float64)\
                                                  ,fundamental_mats[source_ind])
                    if q_value<q_temp:
                        q_value = q_temp
                        max_q = id_q
                #update I, fc , Fg
                final_cand_point = Am_good_x_hat[max_q]
                final_fc = lab_images[Am[max_q]+1][final_cand_point[0],final_cand_point[1],:]
                final_fg = Extracted_feature[Am[max_q]][(final_cand_point[0], final_cand_point[1])]
                #패치를 대체합니다.
                return_img[i,j] = saved_imgs[Am[max_q]+1][final_cand_point[0],final_cand_point[1],:]
                change_patch(saved_imgs[Am[max_q]+1],final_cand_point[0],final_cand_point[1],return_img,i,j,patchsize-3)
                lab_images[0][i, j] = final_fc
                Extracted_feature[0][(i, j)] = final_fg

                #feature mapping과 c를 수정합니다.
                Am_lists = np.array(list(Am))
                list_q_in_AM = np.where(Am_lists ==Am[max_q])
                if list_q_in_AM[0].shape[0] ==1:

                    correspondence_map[Am[max_q]][i, j] = np.flip(final_cand_point-np.array([i,j]))#현재 다른차원에서 매핑된 점을 사용합니다.
                    f_c_x_r = lab_images[0][i, j, :]  # ref이미지에 대한 LAB벡터입니다.
                    f_c_x_hat = lab_images[Am[max_q] + 1][final_cand_point[0], final_cand_point[1], :]  # source 이미지에 대한 LAB벡터입니다.

                    f_g_x_r = Extracted_feature[0][(i, j)]  # ref이미지에 대한 dense sift입니다.
                    f_g_x_hat = Extracted_feature[Am[max_q]][(final_cand_point[0], final_cand_point[1])]  # ref이미지에 대한 dense sift입니다.

                    dist_1 = Se_distance(f_c_x_r, f_c_x_hat, sigma_c)
                    dist_2 = Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                    dist_3 = Sf_distance(np.float64([i, j]), np.float64([final_cand_point[0], final_cand_point[1]]), fundamental_mats[Am[max_q]])
                    if np.isnan(dist_3):
                        print([i, j], [final_cand_point[0], final_cand_point[1]])
                        print("왜 3난이 나올까")
                        exit(0)
                    similarity_map[i, j, Am[max_q]] = Lambda_1 * dist_1 + \
                                                Lambda_2 * dist_2 + \
                                                Lambda_3 * dist_3
                elif list_q_in_AM[0].shape[0] ==2:
                    best_cand_points = [Am_good_x_hat[list_q_in_AM[0][0]],Am_good_x_hat[list_q_in_AM[0][1]]]
                    best_sim =0
                    best_k = -1
                    for best_cand_k, best_cand_point in enumerate(best_cand_points):
                        f_c_x_r = lab_images[0][i, j, :]  # ref이미지에 대한 LAB벡터입니다.
                        f_c_x_hat = lab_images[Am[max_q] + 1][best_cand_point[0], best_cand_point[1], :]  # source 이미지에 대한 LAB벡터입니다.

                        f_g_x_r = Extracted_feature[0][(i, j)]  # ref이미지에 대한 dense sift입니다.
                        f_g_x_hat = Extracted_feature[Am[max_q]][(best_cand_point[0], best_cand_point[1])]  # ref이미지에 대한 dense sift입니다.

                        dist_1 = Se_distance(f_c_x_r, f_c_x_hat, sigma_c)
                        if np.isnan(dist_1):
                            print("왜 1난이 나올까")
                            exit(0)
                        dist_2 = Se_distance(f_g_x_r, f_g_x_hat, sigma_g)
                        if np.isnan(dist_2):
                            print("왜 2난이 나올까")
                            exit(0)
                        dist_3 = Sf_distance(np.float64([i, j]), np.float64([best_cand_point[0], best_cand_point[1]]), fundamental_mats[Am[max_q]])
                        if np.isnan(dist_3):
                            print([i, j], [best_cand_point[0], best_cand_point[1]])
                            print("왜 3난이 나올까")
                            exit(0)
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
                        exit(0)
                    similarity_map[i, j, Am[max_q]] = best_sim
                    correspondence_map[Am[max_q]][i, j] = np.flip(best_cand_points[best_k] - np.array([i, j]))  # 현재 다른차원에서 매핑된 점을 사용합니다.
                else:
                    print("z")
                    exit(0)
                cv2.imshow("mask", mask_img)
                cv2.imshow("result", return_img)
                cv2.waitKey(1)
cv2.waitKey(0)