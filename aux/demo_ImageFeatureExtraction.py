import cv2
import matplotlib
import matplotlib.pyplot as plt

imagePath = 'solar08.jpeg'
image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

cv2.imshow("Resultado", image)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()