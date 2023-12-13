import json

import cv2
import numpy as np
import torch

video_number = 19
video = cv2.VideoCapture(f"../data/test/video_{video_number}.mp4")
frame_n = 0

with open('../data/polygons.json', 'r') as f:
    all_polygons = json.load(f)

polygons = np.array(all_polygons[f'video_{video_number}.mp4'], dtype=np.int32)

with open('../data/time_intervals.json', 'r') as f:
    time_intervals = json.load(f)

time_interval = time_intervals[f'video_{video_number}.mp4']

labels = torch.zeros(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), dtype=torch.float32)
for start, end in time_interval:
    labels[start:end + 1] = 1

annotated_frames = []
while True:
    ret, frame = video.read()
    frame_n += 1
    print(frame_n)
    if not ret:
        break

    if labels[frame_n - 1] == 1:
        cv2.putText(frame, 'Vehicle', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No vehicle', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.polylines(frame, [polygons], True, (0, 255, 0), 2)
    annotated_frames.append(frame)

video.release()

output_video = cv2.VideoWriter('output_annotation.avi', cv2.VideoWriter_fourcc(*'XVID'),
                                5, (annotated_frames[0].shape[1], annotated_frames[0].shape[0]))
for frame in annotated_frames:
    output_video.write(frame)
output_video.release()
