import cv2
import numpy as np
import imutils
import time

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
cofThresh = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("model loading...")

camera = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = camera.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # for (x, y, w, h) in face:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow("Face Detection", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break

camera.release()
cv2.destroyAllWindows()
