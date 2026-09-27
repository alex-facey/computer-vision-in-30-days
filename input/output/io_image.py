import os
import cv2

# ------- read image -------
image_path = os.path.join('', 'data', '001.jpg')

img = cv2.imread(image_path)

# ------- write image -------

cv2.imwrite(os.path.join('', 'data', '001_out.jpg'), img) # output

# ------- visualize image -------

cv2.imshow('image', img)
cv2.waitKey(0) # always needed , number of seconds to keep open

