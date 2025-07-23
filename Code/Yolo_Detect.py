# Import packages
import cv2
from ultralytics import YOLO

# Load YOLOv8m model
model = YOLO("yolov8m.pt")

# Run model on vid.mp4 in streaming mode
results = model("vid.mp4", stream=True)

# Loop through each frame's detection result
for result in results:
    # Get copy of original frame and extract boxes and classes
    frame = result.orig_img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Look through each paired box and class
    for box, cls in zip(boxes, classes):
        # Draws boxes on current frame in pop-up window
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = model.names[int(cls)]
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Breaks the loop if "q" is pressed
    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Destroys window
cv2.destroyAllWindows()
