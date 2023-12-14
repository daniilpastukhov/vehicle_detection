import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import EfficientNet_V2_S_Weights

from motion_detection.models import build_efficientnet_v2_s


def parse_args():
    parser = argparse.ArgumentParser(description="Process video, polygon, and output paths.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("polygon_path", type=str, help="Path to the polygons.")
    parser.add_argument("output_json", type=str, help="Path to the output json file.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = build_efficientnet_v2_s(num_classes=1)

    project_dir = Path(__file__).parent.parent
    ckpt_path = project_dir / 'checkpoints' / 'best_model.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    video = cv2.VideoCapture(args.video_path)

    all_polygons = json.load(Path(args.polygon_path).open())
    polygon = np.array(all_polygons[Path(args.video_path).name], dtype=np.int32)
    bbox = np.array([polygon[:, 0].min(), polygon[:, 1].min(),
                     polygon[:, 0].max(), polygon[:, 1].max()]).reshape(1, 4)

    annotated_frames = []
    raw_preds = []  # model predictions before postprocessing
    preds = []  # model predictions after postprocessing
    while True:
        ret, frame = video.read()
        if not ret:
            break

        cropped_frame = frame[bbox[0, 1]:bbox[0, 3], bbox[0, 0]:bbox[0, 2], :].copy()
        # fill all regions outside polygon with black color
        mask = np.zeros(cropped_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon - bbox[0, :2]], (255, 255, 255))
        cropped_frame[mask == 0, :] = 0
        frame_tensor = torch.tensor(cropped_frame, dtype=torch.float32)
        frame_tensor /= 255
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = frame_tensor.unsqueeze(0)
        frame_tensor = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()(frame_tensor)

        pred = torch.sigmoid(model(frame_tensor)).item()
        is_vehicle = pred > 0.5
        raw_preds.append(is_vehicle)

        # if 3 or more frames in a row are classified as vehicle, then keep it as vehicle
        # to avoid flickering of the label
        if not is_vehicle and sum(raw_preds[-5:]) > 3:
            is_vehicle = True

        preds.append(is_vehicle)
        if is_vehicle:
            cv2.polylines(frame, [polygon], True, (0, 0, 255), thickness=2)
        else:
            cv2.polylines(frame, [polygon], True, (0, 255, 0), thickness=2)
        annotated_frames.append(frame)

    # convert to list of intervals, [0, 0, 1, 1, 1, 0, 0] -> [(2, 4)]
    i = 0
    intervals = []
    for j in range(1, len(preds)):
        if preds[j] != preds[j - 1]:
            if preds[j - 1] == 1:
                intervals.append((i, j - 1))
            i = j
    if i != len(preds) - 1:
        intervals.append((i, len(preds) - 1))

    with open(args.output_json, 'w') as f:
        json.dump({Path(args.video_path).name: intervals}, f)

    video.release()

    # save annotated video, uncomment if needed
    output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 5,
                                   (annotated_frames[-1].shape[1], annotated_frames[-1].shape[0]))
    for frame in annotated_frames:
        output_video.write(frame)
    output_video.release()
