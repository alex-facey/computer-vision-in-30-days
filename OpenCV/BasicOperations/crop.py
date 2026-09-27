# crop
import os
import cv2

img = cv2.imread(os.path.join('.', 'data', '001.jpg'))

print(img.shape)

cropped_img = img[20:350,140:420]

cv2.imshow('image', img)
cv2.imshow('cropped image', cropped_img)
cv2.waitKey(0)