import cv2
import json
import shutil
from datetime import timedelta
from config.settings import (
    VIDEO_PATH, YOLO_MODEL, LOG_JSON, SAVE_DIR, 
    START_TIME, DEBUG_MODE, FRAME_LOOKAHEAD
)
from detector.person_detector import PersonDetector
from detector.reid_tracker import ReIDTracker
from utils.visualization import draw_debug_frame

def main():
    """Main function to run the camera entry/exit tracker"""
    # Initialize components
    person_detector = PersonDetector(YOLO_MODEL)
    reid_tracker = ReIDTracker()
    
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize tracking variables
    logs = []
    frame_buffer = []
    overlap_active = False
    cooldown_counter = 0
    COOLDOWN_FRAMES = int(fps * 1.5)  # 1.5 seconds cooldown
    frame_idx = 0
    
    print("🚀 Starting camera entry/exit tracker...")
    print(f"📹 Processing video: {VIDEO_PATH}")
    print(f"🎯 Door box: {person_detector.door_box}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_time = START_TIME + timedelta(seconds=(frame_idx / fps))
        timestamp = current_time.strftime("%H:%M:%S")

        # Add frame to buffer
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > FRAME_LOOKAHEAD:
            frame_buffer.pop(0)

        # Detect persons in current frame
        person_boxes = person_detector.detect_persons(frame)
        door_overlap_detected, overlapping_person = person_detector.check_door_overlap(person_boxes)

        # Handle door overlap detection
        if door_overlap_detected and not overlap_active and cooldown_counter == 0:
            overlap_active = True
            x1, y1, x2, y2 = overlapping_person
            person_crop = frame[y1:y2, x1:x2]
            
            # Save temporary image
            temp_img_name = f"{SAVE_DIR}/temp_{timestamp}.jpg"
            cv2.imwrite(temp_img_name, person_crop)

            # Track person through frame buffer
            status = reid_tracker.track_person(person_crop, frame_buffer, fps, person_detector)
            
            # Save final image
            final_img_name = f"{SAVE_DIR}/{status}/person_{timestamp}.jpg"
            shutil.move(temp_img_name, final_img_name)
            
            # Log the event
            logs.append({"status": status, "time": timestamp, "img": final_img_name})
            cooldown_counter = COOLDOWN_FRAMES
            
            print(f"👤 {status.upper()} detected at {timestamp}")

        # Update cooldown and overlap state
        if cooldown_counter > 0:
            cooldown_counter -= 1
        if not door_overlap_detected:
            overlap_active = False

        # Debug visualization
        if DEBUG_MODE:
            dbg = draw_debug_frame(frame, person_detector.door_box, person_boxes, 
                                 frame_idx, timestamp, door_overlap_detected)
            cv2.imshow("Debug Frame", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    if DEBUG_MODE:
        cv2.destroyAllWindows()

    # Save logs
    with open(LOG_JSON, "w") as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n📁 Log saved to {LOG_JSON}")
    print(f"📊 Total events detected: {len(logs)}")
    print("✅ Camera tracking completed!")

if __name__ == "__main__":
    main() 