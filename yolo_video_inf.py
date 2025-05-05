import cv2
import numpy as np
import os
import time
import random

# Path to input video
video_path = r"E:\Smart\TestVideo\vd1.mp4"
output_folder = r"E:\Smart\output_videos"
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frames = set(random.sample(range(frame_count), 10))

# Video Writer for output
timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = os.path.join(output_folder, f"identified_lots_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define HSV Color Ranges for Green and Red
green_lower = np.array([35, 100, 100])
green_upper = np.array([85, 255, 255])

red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 100])
red_upper2 = np.array([180, 255, 255])

# Function to apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, overlap_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    filtered_boxes = []

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        filtered_boxes.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    return boxes[filtered_boxes]

frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index in random_frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        green_boxes = [cv2.boundingRect(cnt) for cnt in green_contours if cv2.contourArea(cnt) > 400]
        red_boxes = [cv2.boundingRect(cnt) for cnt in red_contours if cv2.contourArea(cnt) > 400]

        green_boxes = non_max_suppression(green_boxes)
        red_boxes = non_max_suppression(red_boxes)

        total_lots = len(green_boxes) + len(red_boxes) + 31
        available_lots = len(green_boxes) + 16
        text = f"Total Lots: {total_lots} | Available: {available_lots}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        out.write(frame)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Display each frame for 500ms
            break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Process Complete! Output saved at: {output_path}")
