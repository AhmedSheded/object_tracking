import cv2
import numpy as np
from object_detection import ObjectDetection


detection = ObjectDetection()

cap = cv2.VideoCapture('traffic.mp4')

count = 0
center_pts_pre_frame = []

while cap.isOpened():
    ret, frame = cap.read()
    count+=1
    if not ret:
        break
    # points current frame
    center_pts_cur_frame = []

    class_ids, scores, boxes = detection.detect(frame)

    for box in boxes:
        x, y, w, h = box
        cx = int((x+x+w)/2)
        cy = int((y+y+h)/2)
        center_pts_cur_frame.append((cx, cy))


        # print('Frame Nnumber ',count, (x, y, w, h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 250, 200), 2)
    for pt in center_pts_cur_frame:
        cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    cv2.imshow('frame', frame)
    # copy points
    center_pts_pre_frame = center_pts_cur_frame.copy()

    key=cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()