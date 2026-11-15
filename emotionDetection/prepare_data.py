from utils import get_face_landmarks
import os
import cv2
import numpy as np

data_dir = "./data"
output = []

for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path)
        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1434:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))