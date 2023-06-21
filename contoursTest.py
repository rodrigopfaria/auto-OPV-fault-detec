import cv2
import numpy as np

img = cv2.imread('images/inputs/raw/ok/input_ok_1.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY)
cv2.imshow('binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('input_1_thresh_180.jpg', thresh)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
img_copy = thresh.copy()
cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('None approximation', img_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image2.jpg', img_copy)
cv2.destroyAllWindows()
