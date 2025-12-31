import cv2
import torch
import numpy as np
from PIL import Image
import torchreid
from torchvision import transforms
from config.settings import DEVICE, MODEL_NAME, EBD_SIMILARITY_THRESHOLD

class ReIDTracker:
    def __init__(self):
        """Initialize the ReID tracker with ReID model"""
        print("Initializing ResNet50 model...")
        self.reid_model = torchreid.models.build_model(
            name=MODEL_NAME,
            num_classes=0,
            pretrained=True
        ).to(DEVICE).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
    
    def get_embedding(self, crop):
        """Extract embedding from a person crop"""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensor = self.transform(crop_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.reid_model(tensor).cpu().detach().numpy()[0]
        return emb
    
    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    
    def get_adaptive_threshold(self, time_gap_seconds):
        """Get adaptive threshold based on time gap"""
        base_threshold = EBD_SIMILARITY_THRESHOLD
        if time_gap_seconds < 1.0:
            return base_threshold
        elif time_gap_seconds < 5.0:
            return base_threshold - 0.05
        else:
            return base_threshold - 0.1
    
    def track_person(self, person_crop, frame_buffer, fps, person_detector):
        """Track a person through frame buffer to determine entry/exit"""
        person_emb = self.get_embedding(person_crop)
        
        match_count = not_match_count = 0
        for idx, past_frame in enumerate(frame_buffer[:-1]):
            time_gap = (len(frame_buffer) - idx) / fps
            threshold = self.get_adaptive_threshold(time_gap)
            
            # Detect persons in past frame
            past_person_boxes = person_detector.detect_persons(past_frame)
            
            for pbox in past_person_boxes:
                px1, py1, px2, py2 = pbox
                crop = past_frame[py1:py2, px1:px2]
                emb = self.get_embedding(crop)
                if self.cosine_similarity(person_emb, emb) > threshold:
                    match_count += 1
                    break
            else:
                not_match_count += 1
        print(f"Match count: {match_count}, Not match count: {not_match_count}")
        return "exit" if match_count > not_match_count else "entry" 