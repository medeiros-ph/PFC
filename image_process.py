#https://towardsdatascience.com/bag-of-visual-words-in-a-nutshell-9ceea97ce0fb

import cv2
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans 

#Load image to system
image = cv2.imread('images/Solar/solar.jpeg')
#Load image as Grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#defining feature extractor that we want to use
#in this case create SIFT Feature Detector object
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

print("Number of Keypoints Detected: ", len(keypoints))

# Draw rich key points on input image
image = cv2.drawKeypoints(image, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#image_name = "solar01"
#cv2.imwrite("images/copy-SIFT, image)

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

cv2.waitKey(0)
cv2.destroyAllWindows()
