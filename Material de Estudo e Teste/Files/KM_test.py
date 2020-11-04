### Links úteis:
### https://www.hackevolve.com/create-your-own-object-detector/
### https://github.com/saideeptalari/Object-Detector
### https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/
### 


from requests import exceptions
import requests
import dlib
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import imutils
import datetime
import time
import selectors
import sys
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
from sklearn import svm
from sklearn import datasets

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def features(image, sift):
   keypoints, descriptors = sift.detectAndCompute(image, None)
   return keypoints, descriptors

caminho1 = '/home/medeiros/images/solar01.jpeg'
caminho2 = '/home/medeiros/images/solar02.jpeg'
caminho3 = '/home/medeiros/images/solar03.jpeg'
caminho4 = '/home/medeiros/images/solar04.jpeg'
caminho5 = '/home/medeiros/images/solar05.jpeg'
caminho6 = '/home/medeiros/images/solar06.jpeg'
caminho7 = '/home/medeiros/images/solar07.jpeg'
caminho8 = '/home/medeiros/images/solar08.jpeg'
caminho9 = '/home/medeiros/images/solar09.jpeg'
caminho10 = '/home/medeiros/images/solar10.jpeg'
caminho11 = '/home/medeiros/images/solar11.jpeg'
caminho12 = '/home/medeiros/images/solar12.jpeg'
caminho13 = '/home/medeiros/images/solar13.jpeg'
caminho14 = '/home/medeiros/images/solar14.jpeg'
caminho15 = '/home/medeiros/images/solar15.jpeg'
caminho16 = '/home/medeiros/images/solar16.jpeg'
caminho17 = '/home/medeiros/images/solar17.jpeg'
caminho18 = '/home/medeiros/images/solar18.jpeg'
caminho19 = '/home/medeiros/images/solar19.jpeg'
caminho20 = '/home/medeiros/images/solar20.jpeg'
caminho21 = '/home/medeiros/images/solar21.jpeg'
caminho22 = '/home/medeiros/images/solar22.jpeg'
caminho23 = '/home/medeiros/images/solar23.jpeg'
caminho24 = '/home/medeiros/images/solar24.jpeg'
caminho25 = '/home/medeiros/images/solar25.jpeg'
caminho26 = '/home/medeiros/images/solar26.jpeg'

#Selecionar Imagem para teste
caminho = caminho10

print("[INFO] loading image to Grayscale...")
#image = cv2.imread(caminho)
#caminho = imutils.resize(caminho, width=400)
gray = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

print("[INFO] Start Sift...")
sift = cv2.xfeatures2d.SIFT_create()

keypoints, sift = sift.detectAndCompute(gray, None)

print ('[INFO] Number of keypoints Detected: ', len(keypoints))

image = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kmeans = KMeans(n_clusters = 800)
kmeans.fit(sift)

print ('[INFO] KMeans PASSED!')

#preprocessed_image = []
if (sift is not None):
    histogram = build_histogram(sift, kmeans)
    #preprocessed_image.append(histogram)

print ('[INFO] Preprocessed PASSED!')




#Opcao 2

#data = cv2.imread(image)
# data = gray
# keypoints, sift = features(data, sift)
# histogram = build_histogram(sift, kmeans)

#neighbor = NearestNeighbors(n_neighbors = 20)
#neighbor.fit(histogram)
#dist, result = neighbor.kneighbors([histogram])



cv2.imshow("Sift WOrking", image)
cv2.waitKey(0)  & 0xFF
cv2.destroyAllWindows()
