# Vehicle detection

Problem statement:
- Detect whether there is a vehicle inside the region of interest (defined by a polygon).

Implemented solution - EffNet V2 (small) trained as a binary classifier (0 indicates no vehicle, 1 otherwise).

## Getting started
- Create a virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
- Place data in desired location (e.g., `data/`).
- Use `infer.sh` for inference on a single video:
```
chmod +x infer.sh
./infer.sh path_to_video.mp4 polygons.json output.json
```

For training, you can use `train_effnet.py` script.
Make sure to specify the correct paths to the dataset, polygons, and time intervals in the script.

## Dataset
- Splited into two parts: train (75%) and validation (25%) sets, or 15 and 5 videos respectively.
- Each frame was resized to 384x384 pixels.
- The region of interest was cropped from each frame using its bounding box, then area outside polygon was masked. Please refer to [this notebook](notebooks/data_overview.ipynb) for visualizations.

## Training and evaluation
- I've decided to use a pretrained model (EffNet V2 small) and fine-tune it. This model should be sufficient for the task, also we can benefit from transfer learning.
- Loss function: binary cross-entropy - common choice for binary classification. Optimizer: Adam with cosine annealing scheduler, which is suitable for transfer learning.
- Hyperparameters were chosen empirically. Different values were tested and the best ones were chosen.
- The final model was obtained after 11 epochs. It was chosen based on the validation loss.
- The overall accuracy on the validation set is 93%.
- The predictions for consecutive frames are smoothed using a majority vote (if at least 3 out of 5 previous predictions were positive - the current prediction is also positive). This helps to reduce the number of false positives.

## Possible improvements
- We can detect vehicles in the whole frame. After obtaining bounding boxes, we can check if the vehicle is inside the polygon or not. This approach should be more robust, especially when dealing with vehicles that are partially outside the polygon. YOLOv8 (or similar) might be suitable for this task even without fine-tuning (if trained on COCO/etc).
- We can preprocess the frames in a different way, e.g., expand polygon by a little margin, or try to train the model on unmasked frames, and then apply the mask during inference.  
