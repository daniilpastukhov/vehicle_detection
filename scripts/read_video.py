import json

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def build_model():
    num_classes = 1
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # replace 1000 classes classifier with 1 class classifier (for binary classification)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.LazyLinear(num_classes),
    )

    return model


model = build_model()
ckpt_path = '../checkpoints/20231212-22-24-38/best_model.pth'
model.load_state_dict(torch.load(ckpt_path))
model.eval()

video_number = 18
video = cv2.VideoCapture(f"../data/test/video_{video_number}.mp4")
frame_n = 0

with open('../data/polygons.json', 'r') as f:
    all_polygons = json.load(f)

polygon = np.array(all_polygons[f'video_{video_number}.mp4'], dtype=np.int32)
bbox = np.array([polygon[:, 0].min(),
                 polygon[:, 1].min(),
                 polygon[:, 0].max(),
                 polygon[:, 1].max()]).reshape(1, 4)
frames = []
while True:
    ret, frame = video.read()
    if not ret:
        break
    frames.append(frame)
video.release()

annotated_frames = []
for frame in frames:
    cropped_frame = frame[bbox[0, 1]:bbox[0, 3], bbox[0, 0]:bbox[0, 2], :]
    # fill all regions outside polygon with black color
    mask = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon - bbox[0, :2]], (255, 255, 255))
    cropped_frame[mask == 0, :] = 0
    frame_tensor = torch.tensor(cropped_frame, dtype=torch.float32)
    frame_tensor /= 255
    frame_tensor = frame_tensor.permute(2, 0, 1)
    frame_tensor = frame_tensor.unsqueeze(0)
    frame_tensor = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()(frame_tensor)
    pred = model(frame_tensor)
    print(pred, torch.sigmoid(pred))
    pred = torch.sigmoid(pred).item()
    is_vehicle = pred > 0.5
    if is_vehicle:
        cv2.putText(frame, 'Vehicle', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'No vehicle', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
    annotated_frames.append(frame)
    # cv2.imshow('frame2', frame)
    # if cv2.waitKey(66) & 0xFF == ord('q'):
    #     break

output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 5, (frame.shape[1], frame.shape[0]))
for frame in annotated_frames:
    output_video.write(frame)
output_video.release()
