# WebCam Object Scanner

A real-time object detection and hand tracking application using YOLOv3, MediaPipe, and OpenCV. This application uses your webcam to detect objects and track hand gestures in real-time.

## Features
- Real-time object detection using YOLOv3
- Hand tracking and gesture recognition
- Automatic model download on first run
- Displays bounding boxes and confidence scores
- Supports all COCO dataset classes
- Real-time FPS display
- Screenshot capability

## Requirements
- Python 3.7+
- Webcam
- Internet connection (for first-time model download)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Joel04kayy/WebCamScanner.git
cd WebCamScanner
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python object_scanner.py
```

- Press 'q' to quit the application
- The program will automatically download the YOLOv3 model on first run

## Controls
- Press 'q' to quit
- Press 's' to save the current frame
- Press 'c' to toggle confidence threshold

## Hand Gestures
The application recognizes the following hand gestures:
- Open Hand: All fingers up
- Closed Fist: All fingers down
- Peace Sign: Index and middle fingers up
- Pointing: Only index finger up
- Gun Sign: Thumb and index fingers up
- Four Fingers: All fingers up except thumb

## Model Information
- Uses YOLOv3 pre-trained on COCO dataset
- Uses MediaPipe for hand tracking
- Model files are automatically downloaded on first run
- Supports 80 different object classes 