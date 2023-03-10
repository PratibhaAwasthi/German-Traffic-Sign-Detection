# Project: Traffic Sign Detetection and Identification

## Required python libraries

- TensorFlow
- Numpy
- OpenCV
- MatplotLib
- Sklearn
- PIL

## Folder structure

The source code consists of following folders and files:

- Classification_model: This includes dataset and Jupyter Notebook used for creating traffic sign classfication model
- yolo_v4: This folder includes yolo4 training Jupyter Notebook
- images/input_images: input images for traffic sign detection and classification
- images/output_images: output images from traffic sign detection and classification model
- traffic_signs_detect_from_image.py: Python program to classify images present images/input_images and writes output images to images/output_images
- traffic_signs_detect_from_video.py: Python program that takes an input video and creates an output video that has traffic sign classified

## Training Traffic Sign Classification CNN model (One time Execution)

1. Navigate to "Classification_model" directory
2. Run the "classification_cnn.ipynb" Jupyter Notebook
3. Step2 generates "German_traffic_sign.h5" model

## Training YOLO (One time Execution)

1. Navigate to "yolo_v4" directory
2. Run "yolo.ipynb" Jupyter Notebook
3. Generates yolov4_training_last.weights and yolov4_training.cfg files that can be used for identification and classification.

### NOTE: You can use the "German_traffic_sign.h5", "yolov4_training_last.weights" and "yolov4_training.cfg" files that are already created when we trained the model instead of performing above two steps again.

## Execution:

### To classify traffic signs in images

- Place the input images in the "input/input_images" directory
- Run "traffic_signs_detect_from_image.py" program
- Classified result images can be viewed in "input/output_images" directory

### To classify input videos

- Run "traffic_signs_detect_from_video.py" with the input video as an input
  Example command: traffic_signs_detect_from_image.py input.mp4
- Generated output video with the traffic sign classified can be seen in the same directory with the name "input_output.avi" format
