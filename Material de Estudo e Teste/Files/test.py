import cv2
import numpy as np

image = cv2.imread('PFC/images/solar01.jpeg', cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

keypoints, sift = sift.detectAndCompute(image, None)

print ('Number of keypoints Detected: ', len(keypoints))

image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("SIFT WOrking", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
