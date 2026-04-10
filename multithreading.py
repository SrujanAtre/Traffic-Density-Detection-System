from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
import cv2
import numpy as np
import os

def get_density(count):
    if count <= 5:
        return "LOW"
    elif count <= 15:
        return "MEDIUM"
    else:
        return "HIGH"

def countVehicles(video_file):

    ct = CentroidTracker(maxDisappeared=5, maxDistance=50)
    trackableObjects = {}
    total = 0

    video_path = os.getcwd() + video_file
    video_name = os.path.basename(video_path)

    if not os.path.exists(video_path):
        print("Video file not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    while True:
        ret, frame = cap.read()

        # LOOP VIDEO
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        output_img = frame.copy()

        # Foreground mask
        fgMask = backSub.apply(frame)

        # Threshold
        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)

        # Remove noise (morphological operation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Ignore small noise
            if area < 1500:
                continue

            (x, y, w, h) = cv2.boundingRect(cnt)
            rects.append((x, y, x + w, y + h))

            # Draw bounding box
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Track objects
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

                if not to.counted:
                    total += 1
                    to.counted = True

            trackableObjects[objectID] = to

            # Draw ID
            cv2.putText(output_img, f"ID {objectID}",
                        (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            cv2.circle(output_img, (centroid[0], centroid[1]), 4, (255,0,0), -1)

        # Density
        density = get_density(total)

        # Display info
        cv2.putText(output_img, f"Total Vehicles: {total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(output_img, f"Density: {density}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Show video
        cv2.imshow(video_name, output_img)

        # Controls
        key = cv2.waitKey(30) & 0xFF  # change speed here

        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # pause

    cap.release()
    cv2.destroyAllWindows()
    print("Exited")

# Run program
if __name__ == "__main__":
    countVehicles("/videos/3.mp4")
    
