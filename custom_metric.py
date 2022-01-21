import numpy as np
import cv2
def Se_distance(feature1, feature2, numerator_sigma):
    return np.exp(-np.linalg.norm(feature1-feature2)/(2*numerator_sigma*numerator_sigma))

def Sf_distance(point1,point2,fundamental):
    sigma_e =0.17
    sampson_distance = cv2.sampsonDistance(point1,point2,fundamental)
    return np.exp(- sampson_distance/(2*sigma_e*sigma_e))