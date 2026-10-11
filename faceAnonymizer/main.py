import os
import cv2
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_img(img, face_detector):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    out = face_detector.detect(mp_image)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.bounding_box

            # Tasks API bounding_box is already in pixels (not normalized)
            x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # clamp to image bounds so slicing never goes negative / out of range
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x1 + w), min(H, y1 + h)
            if x2 <= x1 or y2 <= y1:
                continue

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

            # blur faces
            img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30))

    return img

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'OpenCV', 'data')

args = argparse.ArgumentParser()

args.add_argument("--mode", default='image')
args.add_argument("--filepath", default=os.path.join(data_dir, 'faceTest.jpg'))

args = args.parse_args()

# detect faces
# blaze_face_short_range is for faces close to the camera and min_detection_confidence is the threshold for detection confidence
model_path = os.path.join(base_dir, 'models', 'blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    min_detection_confidence=0.5,
)

with vision.FaceDetector.create_from_options(options) as face_detector:

    if args.mode.lower() == "image":
        img = cv2.imread(args.filepath)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.filepath}")

        img = process_img(img, face_detector)

        # save image
        out_path = os.path.join(data_dir, 'faceTest_blurred.jpg')
        cv2.imwrite(out_path, img)
        print(f"Saved blurred image to {os.path.abspath(out_path)}")

    elif args.mode.lower() == "video":
        cap = cv2.VideoCapture(args.filepath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.filepath}")

        # use the source video's frame rate and size
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = os.path.join(data_dir, 'faceTest_blurred.mp4')
        output_video = cv2.VideoWriter(out_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detector)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print(f"Saved blurred video to {os.path.abspath(out_path)}")
    
    elif args.mode.lower() == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_img(frame, face_detector)
            cv2.imshow("Face Anonymizer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
