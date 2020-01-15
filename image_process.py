#https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb

import cv2
from sklearn.cluster import kMeans
from sklearn.neighbors import NearestNeighbors

#defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

kmeans = kMeans(n_clusters = 800)
kmeans.fit(descriptor_list)

preprocessed_image = []
for image in images:
    image = gray(image)
    keypoints, descriptor = features(descriptor, kmeans)
    preprocessed_image.append(histogram)

data = cv2.imread(image_path)
data = gray(data)
keypoint, descriptor = features(data, extractor)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors = 20)
neighbor.fit(preprocess_image)
dist, result = neighbor.kneighbors([histogram])
