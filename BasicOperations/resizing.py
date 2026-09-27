import os
import cv2


img = cv2.imread(os.path.join('', 'data', '001.jpg'))

resized_img = cv2.resize(img, (274, 183))

print(img.shape) # (height, width, channel)
print(resized_img.shape) # (height, width, channel)

cv2.imshow('image', img)
cv2.imshow('resized image', resized_img)
cv2.waitKey(0) # always needed , number of seconds to keep open


