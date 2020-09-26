import os
import math
import glob
import time
import cv2
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.feature import hog
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
# %matplotlib inline

print("[INFO] Required packages imported!")

# Hold the color code name and opencv objects in a dict for easy conversion
colorCodeDict = {
    'RGB2GRAY' : cv2.COLOR_RGB2GRAY,
    'RGB2RGBA' : cv2.COLOR_RGB2RGBA,
    'RGB2BGR' : cv2.COLOR_RGB2BGR,
    'RGB2BGRA' : cv2.COLOR_RGB2BGRA,
    'RGB2HSV' : cv2.COLOR_RGB2HSV,
    'RGB2HLS' : cv2.COLOR_RGB2HLS,
    'RGB2LUV' : cv2.COLOR_RGB2LUV,
    'RGB2YUV' : cv2.COLOR_RGB2YUV,
    'RGB2YCrCb' : cv2.COLOR_RGB2YCrCb,
    
    'BGR2GRAY' : cv2.COLOR_BGR2GRAY,
    'BGR2BGRA' : cv2.COLOR_BGR2BGRA,
    'BGR2RGB' : cv2.COLOR_BGR2RGB,
    'BGR2RGBA' : cv2.COLOR_BGR2RGBA,
    'BGR2HSV' : cv2.COLOR_BGR2HSV,
    'BGR2HLS' : cv2.COLOR_BGR2HLS,
    'BGR2LUV' : cv2.COLOR_RGB2LUV,
    'BGR2YUV' : cv2.COLOR_RGB2YUV,
    'BGR2YCrCb' : cv2.COLOR_RGB2YCrCb
}

def convert_color(img, convCode='RGB2GRAY'):
	# Return image converted to required colorspace
	return cv2.cvtColor(img, colorCodeDict[convCode]);

def plot_img(img):
	cv2.imshow("Resultado", img)
	cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()

print("[INFO] Utility functions to convert color space and plot image: Ok!")

def bin_spatial(img, size=(32, 32)):
	# Return the image color bins
	color1 = cv2,resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):
	# Return all channel histogram
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins)
	return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vector=True):
	#Return a histogram of oriented gradients using skimage.
	return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
		cells_per_block=(cell_per_block), transform_sqrt=True,
		visualise=False, feature_vector=feature_vector)	

print("[INFO] Functions to extract different image features: Ok!")
