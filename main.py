import cv2
import numpy as np
import imutils
import time

camera = cv2.VideoCapture(0)

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

