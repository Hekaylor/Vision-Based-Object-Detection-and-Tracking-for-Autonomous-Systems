# Import packages
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Run model on video in streaming mode
results = model("vid.mp4", stream=True)

# OpenCV display
cv2.namedWindow("YOLOv8 + DeepSORT", cv2.WINDOW_NORMAL)

# Logs
position_log = []
decision_log = []
track_history = {}
frame_idx = 0

# Vehicle simulation state
vehicle_position = [0.0, 0.0, 0.0]
vehicle_path = []

# Last track ID
last_seen = defaultdict(lambda: None)
id_switches = 0

# Decision logic
def make_decisions(tracks, frame_width):
    threats = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        # Determine if class name is car, truck, bicycle, or motorbike
        class_name = track.det_class or "unknown"
        if class_name not in ["car", "truck", "bicycle", "motorbike"]:
            continue

        # Define a frame for the label
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x_center = (x1 + x2) / 2
        rel_x = x_center / frame_width
        box_height = y2 - y1
        distance_est = 1000 / (box_height + 1e-5)

        # Determine the recommendation and distance to a threat
        if 0.35 < rel_x < 0.65 and distance_est < 15:
            threats.append(("STEER LEFT", class_name, distance_est, "Object in center path", track))
        elif distance_est < 7:
            threats.append(("BRAKE", class_name, distance_est, "Object close", track))

    if not threats:
        threats.append(("NONE", None, None, "No threat detected", None))

    return threats

# Vehicle update based on decision
def update_position(position, decision):
    x, y, heading = position
    if decision == "BRAKE":
        return (x, y, heading)
    if decision == "STEER LEFT":
        heading += 0.05
    elif decision == "STEER RIGHT":
        heading -= 0.05
    x += 0.5 * np.cos(heading)
    y += 0.5 * np.sin(heading)
    return (x, y, heading)

# Main processing loop
for result in results:
    frame = result.orig_img.copy()
    frame_width = frame.shape[1]

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Prepare detections for DeepSORT
    detections = []
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = box.astype(float)
        bbox = [x1, y1, x2 - x1, y2 - y1]
        class_name = model.names[int(cls_id)]
        detections.append((bbox, float(conf), class_name))

    # Tracker update
    tracks = tracker.update_tracks(detections, frame=frame)

    # Get all threats this frame
    threats = make_decisions(tracks, frame_width)

    # Pick the most urgent threat (first non-NONE)
    active_threat = next((t for t in threats if t[0] != "NONE"), threats[0])
    decision, obj_type, distance, reason, trigger_track = active_threat

    # Simulate vehicle update
    vehicle_position = update_position(vehicle_position, decision)
    vehicle_path.append(vehicle_position)

    # Draw & log all tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        label = track.det_class if track.det_class else "unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Retrieve the object position
        obj_pos = (int((track.to_ltrb()[0] + track.to_ltrb()[2]) / 2),
               int((track.to_ltrb()[1] + track.to_ltrb()[3]) / 2))
        obj_key = (track.det_class, obj_pos)

        # Update ID if it has changed and increment number of changes
        prev_id = last_seen[obj_key]
        if prev_id is not None and prev_id != track.track_id:
            id_switches += 1
        last_seen[obj_key] = track.track_id

        # Save position
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((x_center, y_center))
        track_history[track_id] = track_history[track_id][-30:]

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (255, 255, 0), 2)

        position_log.append([frame_idx, track_id, x_center, y_center])

    # Draw and log all threats
    for threat in threats:
        d, cls, dist, reason, track = threat
        track_id = track.track_id if track else -1
        decision_log.append([frame_idx, d, cls, dist, reason, track_id])

        if track:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(frame, f"{d} ID {track_id}", (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show current decision
    cv2.putText(frame, f"MAIN DECISION: {decision}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display frame
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()

# Save logs
with open("position_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "id", "x", "y"])
    writer.writerows(position_log)

with open("decision_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "decision", "object", "distance", "reason", "track_id"])
    writer.writerows(decision_log)

# Save and plot trajectory
xs, ys = zip(*[(x, y) for x, y, _ in vehicle_path])
plt.plot(xs, ys, marker='o')
plt.title("Simulated Vehicle Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axis("equal")
plt.grid(True)
plt.savefig("vehicle_trajectory.png")
plt.show()

# Create dataframe for decision log
df = pd.read_csv("decision_log.csv")
df = df[df["decision"] != "NONE"]

# Plot distances vs decisions over time
plt.figure(figsize=(10, 6))
for decision in df["decision"].unique():
    subset = df[df["decision"] == decision]
    plt.plot(subset["frame"], subset["distance"], label=decision, marker='o')

plt.title("Object Distance vs. Decision Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Estimated Distance to Object")
plt.legend()
plt.grid(True)
plt.savefig("distance_vs_decision.png")
plt.show()
