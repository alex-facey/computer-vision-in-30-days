import os
import cv2

img = cv2.imread(os.path.join('.', 'data', '001.jpg'))

k_size = 11
blurred_img = cv2.blur(img, (k_size, k_size)) 
gaussian_blurred_img = cv2.GaussianBlur(img, (k_size, k_size), 3)
median_blurred_img = cv2.medianBlur(img, k_size)

cv2.imshow('image', img)
cv2.imshow('blurred image', blurred_img)
cv2.imshow('gaussian blurred image', gaussian_blurred_img)
cv2.imshow('median blurred image', median_blurred_img)
cv2.waitKey(0)