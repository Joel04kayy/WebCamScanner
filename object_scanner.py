import cv2
import numpy as np
import os
import urllib.request

def load_model():
    # Load the pre-trained model and configuration
    model_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    
    # Download the model files if they don't exist
    if not os.path.exists(model_path):
        print("Downloading YOLOv3 weights...")
        # Using a mirror from GitHub
        urllib.request.urlretrieve(
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights",
            model_path
        )
    
    if not os.path.exists(config_path):
        print("Downloading YOLOv3 configuration...")
        # Using the configuration from the darknet repository
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg",
            config_path
        )
    
    # Load the network
    net = cv2.dnn.readNetFromDarknet(config_path, model_path)
    
    # Load class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, classes

def process_frame(frame, net, classes):
    height, width, _ = frame.shape
    
    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set the input to the network
    net.setInput(blob)
    
    # Get the output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Run forward pass
    outputs = net.forward(output_layers)
    
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return boxes, confidences, class_ids, indices

def draw_detections(frame, boxes, confidences, class_ids, indices, classes):
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Loading model...")
    net, classes = load_model()
    print("Model loaded successfully!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        boxes, confidences, class_ids, indices = process_frame(frame, net, classes)
        
        # Draw detections
        frame = draw_detections(frame, boxes, confidences, class_ids, indices, classes)
        
        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 