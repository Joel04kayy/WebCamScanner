# WebCam Object Scanner

This application uses your computer's webcam to detect and identify objects in real-time using computer vision and the YOLOv3 model.

## Setup

1. Make sure you have Python 3.8 or higher installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python object_scanner.py
```

The first time you run the application, it will automatically download the necessary model files:
- YOLOv3 weights (approximately 236MB)
- YOLOv3 configuration file

## Features

- Real-time object detection using your webcam
- Displays bounding boxes around detected objects
- Shows confidence scores for each detection
- Uses YOLOv3 model for accurate object recognition
- Supports detection of 80 different object classes

## Controls

- Press 'q' to quit the application
- Make sure your webcam is connected and accessible

## Note

The model files (*.weights, *.cfg) are downloaded automatically when you first run the application. They are not included in the repository due to their size. 