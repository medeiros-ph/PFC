import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import datasets

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

num1 = -10
num2 = -6

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.0001, C=100)

print(len(digits.data))

x,y = digits.data[:num1], digits.target[:num1]
clf.fit(x,y)

print('Preditction:',clf.predict(digits.data[num2]))
plt.imshow(digits.images[num2], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()