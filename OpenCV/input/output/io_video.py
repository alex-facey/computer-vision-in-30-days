import os
import cv2

# ------- read video -------
video_path = os.path.join('.', 'data', 'boat.mp4')

video = cv2.VideoCapture(video_path)

# ------- visualize vidoe -------
ret = True
while ret: 
    ret, frame = video.read()

    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey(40)

video.release()
cv2.destroyAllWindows()