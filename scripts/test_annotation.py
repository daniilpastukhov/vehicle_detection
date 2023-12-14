import json
from pathlib import Path

import cv2
import numpy as np
import torch

project_dir = Path(__file__).parent.parent

video_number = 19
video_path = project_dir / 'data' / 'test' / f'video_{video_number}.mp4'
video = cv2.VideoCapture(str(video_path))
frame_n = 0

# Read polygon annotation
polygons_path = project_dir / 'data' / 'polygons.json'
with open(str(polygons_path), 'r') as f:
    all_polygons = json.load(f)
polygons = np.array(all_polygons[f'video_{video_number}.mp4'], dtype=np.int32)

time_intervals_path = project_dir / 'data' / 'time_intervals.json'
with open(time_intervals_path, 'r') as f:
    time_intervals = json.load(f)

time_interval = time_intervals[f'video_{video_number}.mp4']

labels = torch.zeros(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), dtype=torch.float32)
for start, end in time_interval:
    labels[start:end + 1] = 1

annotated_frames = []
while True:
    ret, frame = video.read()
    frame_n += 1

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
