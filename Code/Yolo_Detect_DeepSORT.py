# Import packages
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Run model on vid.mp4 in streaming mode
results = model("vid.mp4", stream=True)

# OpenCV window
cv2.namedWindow("YOLOv8 + DeepSORT", cv2.WINDOW_NORMAL)

# Log data
position_log = []
frame_idx = 0

# List of x,y coordinates
track_history = {}

# Loop through each frame's detection result
for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Format detections correctly for DeepSORT
    detections = []
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box.astype(float)
        w = x2 - x1
        h = y2 - y1
        bbox = [x1, y1, w, h]
        confidence = float(conf)
        class_id = int(cls_id)
        class_name = model.names[class_id]
        detections.append((bbox, confidence, class_name))

    # Update tracker
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) 
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Create box and track ID
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Draw bounding box and ID + label
        label = track.det_class if track.det_class else "unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Store current position
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((x_center, y_center))
        if len(track_history[track_id]) > 30:  # limit history length
            track_history[track_id] = track_history[track_id][-30:]

        # Draw path (lines between points)
        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (255, 255, 0), 2)
            
        # Log position for CSV
        position_log.append([frame_idx, track_id, x_center, y_center])


    # Show frame
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    frame_idx += 1

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()

# Save logs
with open("position_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "id", "x", "y"])
    writer.writerows(position_log)
