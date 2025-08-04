#!/usr/bin/env python3
"""
Example usage of the Camera Entry/Exit Tracker

This script demonstrates how to use the modular components
of the camera tracking system.
"""

from config.settings import VIDEO_PATH, YOLO_MODEL, DEBUG_MODE
from detector.person_detector import PersonDetector
from detector.reid_tracker import ReIDTracker
from utils.visualization import draw_debug_frame
import cv2

def example_detection():
    """Example of using the PersonDetector"""
    print("🔍 Example: Person Detection")
    
    # Initialize detector
    detector = PersonDetector(YOLO_MODEL)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video frame")
        return
    
    # Detect persons
    person_boxes = detector.detect_persons(frame)
    print(f"👥 Detected {len(person_boxes)} persons")
    
    # Check door overlap
    overlap_detected, overlapping_person = detector.check_door_overlap(person_boxes)
    print(f"🚪 Door overlap detected: {overlap_detected}")
    
    if DEBUG_MODE:
        # Show debug frame
        debug_frame = draw_debug_frame(frame, detector.door_box, person_boxes, 1, "00:00:01", overlap_detected)
        cv2.imshow("Example Detection", debug_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def example_reid():
    """Example of using the ReIDTracker"""
    print("\n🆔 Example: ReID Tracking")
    
    # Initialize ReID tracker
    reid_tracker = ReIDTracker()
    print("✅ ReID model initialized successfully")

def main():
    """Run example demonstrations"""
    print("🚀 Camera Entry/Exit Tracker - Example Usage")
    print("=" * 50)
    
    # Check if video exists
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in config/settings.py")
        return
    
    # Run examples
    example_detection()
    example_reid()
    
    print("\n✅ Examples completed!")
    print("💡 Run 'python main.py' to start the full tracking system")

if __name__ == "__main__":
    main() 