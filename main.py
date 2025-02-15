import cv2
import numpy as np
import imutils
import time

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
cofThresh = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("model loading...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("model Loaded")

print("starting camera feed")
camera = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = camera.read()
    frame = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    
    #preprocessing image
    imResize = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(cv2.resize(imResize, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    
    detShape = detections.shape[2]
    
    for i in np.arange(0, detShape):    
        confidence = detections[0, 0, i, 2]
        if confidence > cofThresh:
            idx = int(detections[0, 0, i, 1])
            print("classID: ", detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print("boxCoord: ", detections[0, 0, i, 3:7])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break

camera.release()
cv2.destroyAllWindows()
