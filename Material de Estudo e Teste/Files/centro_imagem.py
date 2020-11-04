import cv2
import numpy as np2

pic = cv2.imread('image.jpg')
rows = pic.shape[1]
cols = pic.shape[0]
center = (cols/2, rows/2)

cv2.waitKey(0)
cv2.destroyAllWindows()
