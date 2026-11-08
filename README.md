## Why

I'm interested in computer vision, but the course I wanted was full and the schedule of the other open section didn't line up with mine. I went with this free course instead because I figured I could work on it weekly.

## What

Course: [Learn Computer Vision in 30 Days](https://www.youtube.com/playlist?list=PLb49csYFtO2HAdNGChGzohFJGnJnXBOqd)

Each day of the course is a project. This repo holds my code for each one as I work through them.

## What I Learned
*I'll update this section each time I work on the course.
### Colour Detection
- inRange() makes a mask of the pixels that are inside a certain colour range
- getbbox makes the bounding box
- When theres multiple similar colour objects, it merges into one box 
### Face Anonymizer
- MediaPipe's face detector loads a .tflite model and only keeps faces above a confidence threshold
- One process_img() function handles a single frame, so the same code works for images, videos and  webcam
- argparse lets you pick the mode from the command line (--mode webcam),
### Text Detection
- Compared three OCR tools, Tesseract, EasyOCR, and AWS Textract, on images with short text, using Jaccard similarity (shared words ÷ total unique words) against filename-based ground truth. Tesseract does poorly on photos without preprocessing.
- Learned the Colab workflow: mounting Google Drive, copying and unzipping data into /content, installing system packages with apt, and moving the notebook in and out of a GitHub repo.
- Set up AWS for the first time: created an IAM user and access keys, debugged an AccessDeniedException caused by a service control policy on a new Free plan account, and learned to keep keys out of code and git.
### Image Classification
- Trained an SVM to classify parking spots as empty or not empty: each image is resized to 15x15 and flattened into a 1D array of pixel values to use as features
- GridSearchCV tries every combination of gamma and C with 5-fold cross validation and keeps the best model, which got ~100% on the test set
- Accuracy changes slightly each run because train_test_split shuffles randomly unless you set random_state; saved the trained model with pickle so it doesn't need retraining
### Image Classification with Feature Extraction
- Classified weather images (cloudy, rain, shine, sunrise) by using a pretrained ResNet-18 (img2vec_pytorch) as a feature extractor, turning each image into a 512-number vector, then training a Random Forest on those vectors. Got ~94% accuracy on the validation set
- Split the flat dataset into train/val folders (80/20 per class); the folder names become the labels
- Some images have 4 channels (RGBA), so they need .convert('RGB') before going into the model
- Relative paths like ./data depend on the folder you run from, not where the script is; enabled VS Code's "Execute In File Dir" setting to fix it
- model.p only stores the Random Forest, so infer.py still needs Img2Vec to extract features from new images

## How to Run

Install the requirements:

```
pip install -r requirements.txt
```

Each project folder has its own requirement/instructions for running it.
