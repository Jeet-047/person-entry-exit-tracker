import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config.settings import DEVICE, CONF_THRESHOLD, OVERLAP_THRESHOLD, DOOR_BOX

class PersonDetector:
    def __init__(self, yolo_model_path):
        """Initialize the person detector with YOLO model"""
        self.yolo = YOLO(yolo_model_path).to(DEVICE)
        self.door_box = DOOR_BOX
    
    def compute_iou(self, boxA, boxB):
        """Compute Intersection over Union between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    
    def detect_persons(self, frame):
        """Detect persons in the frame and return bounding boxes"""
        results = self.yolo(frame)[0]
        person_boxes = []
        
        for box in results.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            person_box = (x1, y1, x2, y2)
            person_boxes.append(person_box)
        
        return person_boxes
    
    def check_door_overlap(self, person_boxes):
        """Check if any person overlaps with the door area"""
        door_overlap_detected = False
        overlapping_person = None
        
        for person_box in person_boxes:
            if self.compute_iou(person_box, DOOR_BOX) >= OVERLAP_THRESHOLD:
                door_overlap_detected = True
                overlapping_person = person_box
                break
        
        return door_overlap_detected, overlapping_person 