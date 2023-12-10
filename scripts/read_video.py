import json

import cv2
import numpy as np

video = cv2.VideoCapture("../videos/video_1.mp4")
frame_n = 0

with open('../data/polygons.json', 'r') as f:
    all_polygons = json.load(f)

polygons = np.array(all_polygons['video_1.mp4'], dtype=np.int32)

while True:
    ret, frame = video.read()
    frame_n += 1

    if not ret:
        break

    frame = cv2.polylines(frame, [polygons], True, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(66 * 2) & 0xFF == ord('q'):
        break
