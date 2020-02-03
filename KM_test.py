import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def features(image, sift):
   keypoints, descriptors = sift.detectAndCompute(image, None)
   return keypoints, descriptors

caminho = '/home/medeiros/images/solar01.jpeg'
caminho2 = '/home/medeiros/images/solar02.jpeg'

image = cv2.imread(caminho)
image2 = cv2.imread(caminho2)

gray = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
gray2 = cv2.imread(caminho2, cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, sift = sift.detectAndCompute(image, None)

print ('Number of keypoints Detected: ', len(keypoints))

image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kmeans = KMeans(n_clusters = 800)
kmeans.fit(sift)

print ('KMeans PASSED!')

#preprocessed_image = []
if (sift is not None):
    histogram = build_histogram(sift, kmeans)
    #preprocessed_image.append(histogram)

print ('Preprocessed PASSED!')




#Opcao 2

#data = cv2.imread(image)
# data = gray
# keypoints, sift = features(data, sift)
# histogram = build_histogram(sift, kmeans)
neighbor = NearestNeighbors(n_neighbors = 20)
neighbor.fit(histogram)
dist, result = neighbor.kneighbors([histogram])



cv2.imshow("SVM WOrking", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
