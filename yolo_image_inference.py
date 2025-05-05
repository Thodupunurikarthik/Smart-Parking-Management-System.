import cv2
import torch
import os
import time
from ultralytics import YOLO

# Load trained YOLOv8 model
model_path = r"E:\Smart\runs\detect\train19\weights\best.pt"
model = YOLO(model_path)  

# Correct image path format
image_path = r"E:\Smart\testimg\2013-04-13_09_45_04_jpg.rf.b6686e3b856b841aef493bacca9d691a.jpg"                   
image = cv2.imread(image_path)

# Ensure image is loaded
if image is None:
    print("❌ Error: Image not found. Check the path.")
    exit()

# Run inference with a confidence threshold
results = model.predict(image, conf=0.25)  

# Initialize counters
total_lots = 0
available_lots = 0

# Process detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])  # Class ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates

        # Define colors: Green (empty), Red (occupied)
        color = (0, 255, 0) if cls == 1 else (0, 0, 255)

        # Draw bounding box without text
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Update counters
        total_lots += 1
        if cls == 1:  # Class 1 = Empty space
            available_lots += 1

# Display total and available lots at the top
text = f"Total Lots: {total_lots} | Available: {available_lots}"
cv2.putText(image, text, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 10), 3)

# Create output folder if it doesn't exist
output_folder = r"E:\Smart\output_images"
os.makedirs(output_folder, exist_ok=True)

# Generate a unique filename using timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = os.path.join(output_folder, f"parking_lot_output_{timestamp}.jpg")

# Save the output image
cv2.imwrite(output_path, image)

# Show the image
cv2.imshow("Parking Lot Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Detection complete! Output saved at: {output_path}")