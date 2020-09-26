#!/usr/bin/env

from requests import exceptions
import requests
import dlib
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json 
import glob
#import urllib.request
import argparse
import imutils
import datetime
import time
#import selectors
import sys
import random
import os
from imutils.paths import list_images
from imutils.video import VideoStream
from imutils import face_utils
#from selectors import BoxSelector
#from detector import ObjectDetector

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn import svm
from sklearn.svm import SVC

from sklearn import datasets

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# importing library for plotting 
from matplotlib import pyplot as plt 


def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def features(image, sift):
   keypoints, descriptors = sift.detectAndCompute(image, None)
   return keypoints, descriptors

def centro_imagem(pic):
	rows = pic.shape[1]
	cols = pic.shape[0]
	center = (cols/2, rows/2)
	return center

#def fd_hu_moments(image):
#    feature = cv2.HuMoments(cv2.moments(image)).flatten()
#    return feature

#def fd_haralick(gray):
    # compute the haralick texture feature vector
#    haralick = mahotas.features.haralick(gray).mean(axis=0)
#    return haralick

### https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html 
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [25, 25, 25], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape:",hist.shape)
    print("with: ",hist.flatten().shape[0])
    # normalize the histogram
    cv2.normalize(hist, hist)
    hist.flatten()
    return hist

#caminho01 = '/home/viki/PFC/images/solar01.jpeg'
caminho02 = '/home/medeiros/PFC/images/solar02.jpeg'
caminho03 = '/home/medeiros/PFC/images/solar03.jpeg'
caminho04 = '/home/medeiros/PFC/images/solar04.jpeg'
caminho05 = '/home/medeiros/PFC/images/solar05.jpeg'
caminho06 = '/home/medeiros/PFC/images/solar06.jpeg'
caminho07 = '/home/medeiros/PFC/images/solar07.jpeg'
caminho08 = '/home/medeiros/PFC/images/solar08.jpeg'
caminho09 = '/home/medeiros/PFC/images/solar09.jpeg'
caminho10 = '/home/medeiros/PFC/images/solar10.jpeg'
caminho11 = '/home/medeiros/PFC/images/solar11.jpeg'
caminho12 = '/home/medeiros/PFC/images/solar12.jpeg'
caminho13 = '/home/medeiros/PFC/images/solar13.jpeg'
caminho14 = '/home/medeiros/PFC/images/solar14.jpeg'
caminho15 = '/home/medeiros/PFC/images/solar15.jpeg'
caminho16 = '/home/medeiros/PFC/images/solar16.jpeg'
caminho17 = '/home/medeiros/PFC/images/solar17.jpeg'
caminho18 = '/home/medeiros/PFC/images/solar18.jpeg'
caminho19 = '/home/medeiros/PFC/images/solar19.jpeg'
caminho20 = '/home/medeiros/PFC/images/solar20.jpeg'
caminho21 = '/home/medeiros/PFC/images/solar21.jpeg'
caminho22 = '/home/medeiros/PFC/images/solar22.jpeg'
caminho23 = '/home/medeiros/PFC/images/solar23.jpeg'
caminho24 = '/home/medeiros/PFC/images/solar24.jpeg'
caminho25 = '/home/medeiros/PFC/images/solar25.jpeg'
caminho26 = '/home/medeiros/PFC/images/solar26.jpeg'

#Selecionar Imagem para teste
caminho = caminho02

print("[INFO] Loading image to Grayscale...")
#image = cv2.imread(caminho)
#caminho = imutils.resize(caminho, width=400)
#gray = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(caminho)



#Opcao 2

#data = cv2.imread(image)
# data = gray
# keypoints, sift = features(data, sift)
# histogram = build_histogram(sift, kmeans)

#neighbor = NearestNeighbors(n_neighbors = 20)
#neighbor.fit(histogram)
#dist, result = neighbor.kneighbors([histogram])


#fd_haralick(image) CNN
#fd_hu_moments(image) CNN
print ('[INFO] Concatenate all the features obtained')
global_feature = np.hstack([fd_histogram(img2)])

#remove os NaN do vetor
#global_feature = global_feature[~np.isnan(global_feature)]
global_feature = global_feature[~pd.isnull(global_feature)]

scaler = MinMaxScaler(feature_range=(0, 1))
print ('[INFO] Normalize The feature vectors...')

#modifiquei para o fit e o transform ficarem separados
rescaled_features = scaler.fit(global_feature.reshape(-1,1))
print(scaler.data_max_)
rescaled_features = scaler.transform(global_feature.reshape(-1,1))


print ('[INFO] Running Support Vector Machine')
clf = SVC(random_state=9)

#modifiquei a funcao pois precisa de numero minimo de features
y = np.ones(1)
x = global_feature.reshape(1,-1)

#prediction= clf.fit(global_feature,y)[0]
prediction= clf.fit(global_feature.reshape(1,-1),y)[0]
print ('[INFO] Done Processing')
ci2.imshow("Resultado", image)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
