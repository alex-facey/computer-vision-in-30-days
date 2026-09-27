import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'whiteboard.jpg'))
print(img.shape)

# line
cv2.line(img, (100, 150), (300, 400), (0, 255, 0), 3) # x,y coordinates start to end then color then thickness

# rectangle
cv2.rectangle(img, (100,250), (200, 350), (0,0, 255), -1) #top left corner, bottom right corner

# circle
cv2.circle(img, (200, 350), 15, (255, 170, 50), 10)

# text 
cv2.putText(img, 'Wagwan!', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)