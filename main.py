import cv2
from lane_detection.lane_detector import LaneDetector

# Initialize the lane detector
detector = LaneDetector("pretrained_weights/tusimple_18.pth", dataset="Tusimple")

# Open the input video
cap = cv2.VideoCapture("crash_ori3.mp4")
if not cap.isOpened():
    print("Error: Cannot open input video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes
    lanes = detector.detect_lanes(frame)

    # Draw detected lanes on the frame
    for lane in lanes:
        for point in lane:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Write the annotated frame to the output video
    out.write(frame)

cap.release()
out.release()
print("Lane detection complete. Output saved to output_video.mp4.")
