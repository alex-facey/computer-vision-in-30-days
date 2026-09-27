import os
import cv2
import numpy as np

img = cv2.imread(os.path.join('.', 'data', 'neymar.jpg'))

img_edge = cv2.Canny(img, 300,500) # play around with diff numbers it tells how accurate to get the ddge 

img_edge_d =  cv2.dilate(img_edge, np.ones((5, 5), dtype=np.int8)) #numbers decide thickness

img_edge_e = cv2.erode(img_edge_d, np.ones((5, 5), dtype=np.int8))

cv2.imshow('img', img)
cv2.imshow('img edge', img_edge)
cv2.imshow('img edge dilated', img_edge_d)
cv2.imshow('img edge dilated eroded', img_edge_e)
cv2.waitKey(0)