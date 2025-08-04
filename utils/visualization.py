import cv2
from config.settings import DOOR_BOX, OVERLAP_THRESHOLD

def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def draw_debug_frame(frame, door_box, person_boxes, frame_idx, timestamp, overlap=False):
    """Draw debug frame with door box, person boxes, and debug information"""
    debug_frame = frame.copy()
    x1, y1, x2, y2 = door_box
    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(debug_frame, "DOOR", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for x1, y1, x2, y2 in person_boxes:
        color = (0, 255, 255) if compute_iou((x1, y1, x2, y2), door_box) >= OVERLAP_THRESHOLD else (0, 255, 0)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_frame, "Person", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(debug_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_frame, f"Time: {timestamp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if overlap:
        cv2.putText(debug_frame, "DOOR OVERLAP DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return debug_frame 