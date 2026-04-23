import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

# Video capture
cap = cv2.VideoCapture("video.mp4")

# Line position
line_y = 500
offset = 10

# Tracking
object_id = 0
centers = {}
counted_ids = set()

# Vehicle counts
vehicle_count = {
    "car": 0,
    "motorbike": 0,
    "bus": 0,
    "truck": 0
}

# CSV log
log_data = []

# Allowed classes (YOLO names)
vehicle_classes = ["car", "motorbike", "bus", "truck"]

def get_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = get_center(x1, y1, x2, y2)

        detections.append((cx, cy, label, x1, y1, x2, y2))

    # Tracking (simple centroid matching)
    new_centers = {}

    for cx, cy, label, x1, y1, x2, y2 in detections:
        matched_id = None

        for id, (px, py, plabel) in centers.items():
            if abs(cx - px) < 50 and abs(cy - py) < 50:
                matched_id = id
                break

        if matched_id is None:
            matched_id = object_id
            object_id += 1

        new_centers[matched_id] = (cx, cy, label)

        # Counting logic
        if matched_id not in counted_ids:
            if abs(cy - line_y) < offset:
                vehicle_count[label] += 1
                counted_ids.add(matched_id)

                # Log data
                log_data.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "vehicle": label,
                    "total": vehicle_count[label]
                })

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID:{matched_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    centers = new_centers

    # Draw counting line
    cv2.line(frame, (0, line_y), (1200, line_y), (255, 0, 0), 3)

    # Display counts
    y_offset = 30
    for key, value in vehicle_count.items():
        cv2.putText(frame, f"{key}: {value}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    cv2.imshow("Vehicle Detection System", frame)

    if cv2.waitKey(1) == 27:
        break

# Save CSV
df = pd.DataFrame(log_data)
df.to_csv("vehicle_log.csv", index=False)

cap.release()
cv2.destroyAllWindows()