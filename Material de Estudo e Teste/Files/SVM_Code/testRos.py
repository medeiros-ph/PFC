from detector import ObjectDetector
import numpy as np
import cv2
import argparse
import imutils
import datetime
import time

ap = argparse.ArgumentParser()
ap.add_argument("-d","--detector",required=True,help="path to trained detector to load...")
ap.add_argument("-i","--image",required=True,help="path to an image for object detection...")
ap.add_argument("-a","--annotate",default=None,help="text to annotate...")
args = vars(ap.parse_args())

detector = ObjectDetector(loadPath=args["detector"])

imagePath = args["image"]
image = cv2.imread(imagePath)


video_capture = cv2.VideoCapture('aLine.mp4')
while(True):
	ret, frame = video_capture.read()
	detector.detectsp(frame,annotate=args["annotate"])
	time.sleep(0.001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



detector.detect(image,annotate=args["annotate"])
