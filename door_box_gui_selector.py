import cv2
import os

VIDEO_PATH = "data/input/Camera 1.mp4"  # Replace with your video path

class DoorBoxSelector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.corner_points = []
        self.img_copy = None
        
    def click_corner(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add corner point
            self.corner_points.append((x, y))
            
            # Draw point on image
            cv2.circle(self.img_copy, (x, y), 5, (0, 0, 255), -1)  # Red circle
            cv2.putText(self.img_copy, f"{len(self.corner_points)}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw lines between points
            if len(self.corner_points) > 1:
                for i in range(len(self.corner_points) - 1):
                    cv2.line(self.img_copy, self.corner_points[i], self.corner_points[i+1], 
                            (0, 255, 0), 2)
            
            # Draw line from last point to first point if we have 4 points
            if len(self.corner_points) == 4:
                cv2.line(self.img_copy, self.corner_points[3], self.corner_points[0], 
                        (0, 255, 0), 2)
            
            cv2.imshow("Select Door Corners", self.img_copy)
            
            # Print current point
            print(f"📍 Corner {len(self.corner_points)}: ({x}, {y})")
            
            # If we have 4 points, calculate bounding box
            if len(self.corner_points) == 4:
                self.calculate_bounding_box()
    
    def calculate_bounding_box(self):
        """Calculate bounding box from 4 corner points"""
        if len(self.corner_points) != 4:
            return
        
        # Extract x and y coordinates
        x_coords = [point[0] for point in self.corner_points]
        y_coords = [point[1] for point in self.corner_points]
        
        # Calculate bounding box
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        
        coords = (x1, y1, x2, y2)
        
        print("\n" + "="*50)
        print("🎯 DOOR BOUNDING BOX CALCULATED!")
        print("="*50)
        print(f"📋 Copy this line to your code: DOOR_BOX = {coords}")
        print("="*50)
        
        # Draw bounding box rectangle
        cv2.rectangle(self.img_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue rectangle
        cv2.imshow("Select Door Corners", self.img_copy)
        
        # Wait for user to see the result
        print("Press any key to continue...")
        cv2.waitKey(0)
    
    def select_door_box(self):
        # Load first frame of video
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read first frame from {self.video_path}")
        
        self.img_copy = frame.copy()
        cv2.namedWindow("Select Door Corners", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Door Corners", self.click_corner)
        
        print("🟩 Click 4 corners of the door in any order.")
        print("🟩 Press 'r' to reset, 'q' to quit.")
        cv2.imshow("Select Door Corners", self.img_copy)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("❌ Selection cancelled.")
                cv2.destroyAllWindows()
                return None
                
            elif key == ord('r'):
                # Reset selection
                self.corner_points = []
                self.img_copy = frame.copy()
                cv2.imshow("Select Door Corners", self.img_copy)
                print("🔄 Selection reset. Click 4 corners again.")
        
        cv2.destroyAllWindows()

def main():
    print("🚪 Door Box Selector (4 Corner Points)")
    print("=" * 50)
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script.")
        return
    
    selector = DoorBoxSelector(VIDEO_PATH)
    
    # Select door box
    print(f"📹 Loading video: {VIDEO_PATH}")
    selector.select_door_box()
    
    print("\n✅ Use the coordinates above in your camera_tracker.py file!")

if __name__ == "__main__":
    main()