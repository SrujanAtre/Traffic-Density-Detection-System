from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# VIDEO PATHS (4 directions)
video_paths = [
    "videos/1.mp4",
    "videos/2.mp4",
    "videos/3.mp4",
    "videos/4.mp4",
]

caps = [cv2.VideoCapture(v) for v in video_paths]

frames_to_process = 50
lane_counts = [0, 0, 0, 0]

# PROCESS VIDEOS WITH VISUAL OUTPUT
for _ in range(frames_to_process):
    frames = []

    # Read frames from all videos
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = None
        else:
            frame = cv2.resize(frame, (500, 300))
        frames.append(frame)

    # Process each frame
    for i, frame in enumerate(frames):
        if frame is None:
            continue

        results = model(frame, verbose=False)[0]

        count = 0

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            # Vehicle classes
            if cls in [2, 3, 5, 7]:
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lane_counts[i] += count

        # Show count on each frame
        cv2.putText(frame, f"Lane {i+1}: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Combine 4 frames into one window
    if all(frame is not None for frame in frames):
        top = cv2.hconcat([frames[0], frames[1]])
        bottom = cv2.hconcat([frames[2], frames[3]])
        combined = cv2.vconcat([top, bottom])

        cv2.imshow("Smart Traffic System (YOLOv8)", combined)

    # Press ESC to stop
    if cv2.waitKey(30) & 0xFF == 27:
        break

# AVERAGE COUNTS
lane_counts = [int(c / frames_to_process) for c in lane_counts]

# Release videos
for cap in caps:
    cap.release()

cv2.destroyAllWindows()

# WRITE TO FILE
with open("out.txt", "w") as f:
    for count in lane_counts:
        f.write(str(count) + "\n")

# READ FILE (existing logic)
with open("out.txt", "r") as f:
    no_of_vehicles = [int(f.readline()) for _ in range(4)]

baseTimer = 120
timeLimits = [5, 30]

print("Input no of vehicles :", *no_of_vehicles)

# SAFE TOTAL
total = sum(no_of_vehicles) if sum(no_of_vehicles) != 0 else 1

# SIGNAL TIMING CALCULATION
t = [
    (i / total) * baseTimer
    if timeLimits[0] < (i / total) * baseTimer < timeLimits[1]
    else min(timeLimits, key=lambda x: abs(x - (i / total) * baseTimer))
    for i in no_of_vehicles
]

# OUTPUT
print("\nSignal timings (seconds):")
for i, time in enumerate(t):
    print(f"Lane {i+1}: {time:.2f} sec")

print("Total cycle time:", sum(t))

