import cv2
import os
import time
from ultralytics import YOLO

# Load trained YOLOv8 model
model_path = r"E:\Smart\runs\detect\train19\weights\best.pt"
model = YOLO(model_path)

# Path to input video
video_path = r"E:\Smart\TestVideo\3.mp4"

# Ensure video file exists
if not os.path.exists(video_path):
    print("❌ Error: Video file not found. Check the path.")
    exit()

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output folder if it doesn't exist
output_folder = r"E:\Smart\output_videos"
os.makedirs(output_folder, exist_ok=True)

# Generate unique filename
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_video_path = os.path.join(output_folder, f"parking_lot_output_{timestamp}.mp4")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frame is read

    # Run YOLO inference
    results = model.predict(frame, conf=0.2)  # Lower confidence threshold if needed

    # Initialize counters
    total_lots = 0
    available_lots = 0
    occupied_lots = 0

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates

            # Debugging: Print detected class
            print(f"Detected Class: {cls}")

            # Define colors for visualization
            if cls == 1:  # Class 1 = Empty space
                color = (0, 255, 0)  # Green
                available_lots += 1
            elif cls == 2:  # Class 2 = Occupied space
                color = (0, 0, 255)  # Red
                occupied_lots += 1
            else:
                color = (255, 255, 255)  # Default White (if class unknown)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Update total count
            total_lots += 1

    # Display total and available lots at the top
    text = f"Total Lots: {total_lots} | Available: {available_lots} | Occupied: {occupied_lots}"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Write frame to output video
    out.write(frame)

    # Show video output (disable if running on server)
    cv2.imshow("Parking Lot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video processing complete! Output saved at: {output_video_path}")
