import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

class ParkingSpaceTracker:
    def __init__(self, iou_threshold=0.5, max_missing=5):
        self.spaces = {}
        self.current_id = 0
        self.iou_thresh = iou_threshold
        self.max_missing = max_missing

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area)

    def update(self, detections):
        active_ids = []
        
        # Update existing spaces
        for det in detections:
            max_iou = 0
            best_match = None
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            
            for space_id, data in self.spaces.items():
                iou = self.calculate_iou([x1,y1,x2,y2], data['coords'])
                if iou > max_iou and iou > self.iou_thresh:
                    max_iou = iou
                    best_match = space_id
            
            if best_match is not None:
                self.spaces[best_match]['coords'] = [x1,y1,x2,y2]
                self.spaces[best_match]['status'] = int(det.cls)
                self.spaces[best_match]['missing'] = 0
                active_ids.append(best_match)
            else:
                self.spaces[self.current_id] = {
                    'coords': [x1,y1,x2,y2],
                    'status': int(det.cls),
                    'missing': 0
                }
                active_ids.append(self.current_id)
                self.current_id += 1

        # Handle missing spaces
        for space_id in list(self.spaces.keys()):
            if space_id not in active_ids:
                self.spaces[space_id]['missing'] += 1
                if self.spaces[space_id]['missing'] > self.max_missing:
                    del self.spaces[space_id]

def perspective_transform(frame, src_points, dst_size=(640, 640)):
    h, w = dst_size
    dst_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])
    M = cv2.getPerspectiveTransform(src_points, dst_pts)
    return cv2.warpPerspective(frame, M, (w,h))

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    limg = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def process_video():
    # Configuration
    model_path = r"E:\Smart\runs\detect\train19\weights\best.pt"
    video_path = r"E:\Smart\TestVideo\3.mp4"
    output_folder = r"E:\Smart\output_videos"
    perspective_points = np.float32([[200,300], [800,300], [1000,600], [50,600]])  # Adjust these
    
    # Initialize system
    model = YOLO(model_path)
    tracker = ParkingSpaceTracker()
    os.makedirs(output_folder, exist_ok=True)
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output writer
    output_path = os.path.join(output_folder, f"output_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))
    
    # Processing loop
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing
        enhanced = enhance_contrast(frame)
        warped = perspective_transform(enhanced, perspective_points)
        
        # Detection
        results = model.predict(
            warped,
            conf=0.35,
            imgsz=640,
            augment=True,
            agnostic_nms=True,
            classes=[1,2],  # Only empty/occupied
            verbose=False
        )
        
        # Update tracker
        tracker.update(results[0].boxes)
        
        # Visualization
        available = 0
        occupied = 0
        for space_id, data in tracker.spaces.items():
            x1, y1, x2, y2 = data['coords']
            status = data['status']
            
            color = (0,255,0) if status == 1 else (0,0,255)
            cv2.rectangle(warped, (x1,y1), (x2,y2), color, 2)
            
            if status == 1:
                available += 1
            else:
                occupied += 1
        
        # Status overlay
        cv2.putText(warped, f"Available: {available}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(warped, f"Occupied: {occupied}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        
        # Debug grid
        for i in range(0, 640, 50):
            cv2.line(warped, (i,0), (i,640), (50,50,50), 1)
            cv2.line(warped, (0,i), (640,i), (50,50,50), 1)
        
        # Write output
        out.write(warped)
        cv2.imshow("Parking Monitoring", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete: {output_path}")

if __name__ == "__main__":
    process_video()