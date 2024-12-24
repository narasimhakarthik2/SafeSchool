from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


class WeaponDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.75):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def predict(self, frame: np.ndarray):
        """Run prediction on a single frame"""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold
        )[0]
        return results