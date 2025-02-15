import cv2
import numpy as np
import imutils
import time

def load_model(prototxt_path, model_path):
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("Model loaded successfully")
    return net

def initialize_camera(camera_index=0):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise ValueError("Failed to open camera")
    time.sleep(2.0)
    return camera

def process_frame(frame, net, conf_threshold, classes, colors):
    # Resize frame
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    
    # Preprocess image
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                0.007843, 
                                (300, 300),
                                127.5)
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf_threshold:
            class_id = int(detections[0, 0, i, 1])
            if class_id >= len(classes):
                print(f"Warning: Detected class ID {class_id} is out of range")
                continue
                
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            label = f"{classes[class_id]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                         colors[class_id].tolist(), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id].tolist(), 2)
    
    return frame

def main():
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    CONFIDENCE_THRESHOLD = 0.2
    
    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    
    try:
        net = load_model(prototxt, model)
        camera = initialize_camera()
        
        print("Starting camera feed...")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            processed_frame = process_frame(frame, net, CONFIDENCE_THRESHOLD,
                                         CLASSES, COLORS)
            cv2.imshow("Object Detection", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        print("Cleaning up...")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()