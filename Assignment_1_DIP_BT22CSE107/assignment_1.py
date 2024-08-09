import cv2
import numpy as np


#read image
img = cv2.imread("fruit-salad-12.jpg")

#convert color img to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img", img)                 #display original image using cv2_imshow
cv2.imshow("img_gray", img_gray)       #display image converted to grayscale using cv2_imshow

cv2.waitKey(0)
cv2.destroyAllWindows()