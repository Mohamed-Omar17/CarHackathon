from ultralytics import YOLO
import cv2
import glob
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
carA_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraA'
carB_dir = current_dir / 'Fusion_event-main' / 'data' / 'CameraB'


model = YOLO("yolov8n.pt")


def get_detections_from_image(image_folder):
    all_boxes = []
    all_classes = []
    all_confs = []

    image_paths = glob.glob(f"{image_folder}/*.png")

    if not image_paths:
        print(f"No images found in {image_folder}")

    for img_path in image_paths:
        print(f"Processing {img_path}")  # Debugging line
        results = model(img_path)

        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            all_boxes.append(boxes)
            all_confs.append(confs)
            all_classes.append(classes)
        else:
            print(f"No objects detected in {img_path}")

    # Safely handle empty lists
    if len(all_boxes) == 0:
        print("No detections found.")
        return np.array([]), np.array([]), np.array([])

    return np.vstack(all_boxes), np.hstack(all_classes), np.hstack(all_confs)


boxes_a, classes_a, confs_a = get_detections_from_image(carA_dir)
boxes_b, classes_b, confs_b = get_detections_from_image(carB_dir)