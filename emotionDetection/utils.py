# clean data
# go into data folder and remove inaccurate faces, duplicates, wrong emootions etc

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# same as the tutorial's utils.py, but using the Tasks API
# (mp.solutions.face_mesh doesn't exist in new mediapipe)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'face_landmarker.task')

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def get_face_landmarks(image, draw=False, static_image_mode=True):
    # static_image_mode is kept so the tutorial's calls still work (not needed by the Tasks API)
    image_rows, image_cols, _ = image.shape

    # Kaggle images are 48x48 - too small for mediapipe, so upscale them first
    image_input = image
    if image_rows < 200:
        image_input = cv2.resize(image, (image_cols * 4, image_rows * 4), interpolation=cv2.INTER_CUBIC)

    image_input_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_input_rgb)
    results = _get_landmarker().detect(mp_image)

    image_landmarks = []

    if results.face_landmarks:
        ls_single_face = results.face_landmarks[0]

        if draw:
            for lm in ls_single_face:
                cv2.circle(image, (int(lm.x * image_cols), int(lm.y * image_rows)), 1, (0, 255, 0), -1)

        xs_ = []
        ys_ = []
        zs_ = []
        # convert to pixels so non-square webcam frames don't stretch the face
        for idx in ls_single_face:
            xs_.append(idx.x * image_cols)
            ys_.append(idx.y * image_rows)
            zs_.append(idx.z * image_cols)

        # divide by face width so a small webcam face matches the full-frame Kaggle faces
        scale = (max(xs_) - min(xs_)) or 1.0

        for j in range(len(xs_)):
            image_landmarks.append((xs_[j] - min(xs_)) / scale)
            image_landmarks.append((ys_[j] - min(ys_)) / scale)
            image_landmarks.append((zs_[j] - min(zs_)) / scale)

    return image_landmarks
