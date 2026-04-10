# Traffic Density Detection System

### 1. Overview:

This project implements an intelligent traffic density detection system that dynamically adjusts signal timings based on real-time vehicle density. Two approaches are used:

YOLOv8-based vehicle detection (modern deep learning approach)
Classical image processing with centroid tracking (background subtraction approach)


### 2. Objective:

To detect vehicles from multiple traffic video feeds and compute adaptive traffic signal timings based on vehicle density in each lane.


### 3. Technologies Used:

Python

OpenCV

Ultralytics YOLOv8

NumPy

Centroid Tracking Algorithm


### 4. Project Structure:
Adaptive-Traffic-Signal-Control-System/
│
├── program.py                # YOLOv8-based implementation (main system)
├── multithreading.py        # Classical image processing + tracking
├── tracking/
│   ├── centroidtracker.py
│   └── trackableobject.py
├── videos/
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── 3.mp4
│   ├── 4.mp4
│   └── test.mp4
├── out.txt                  # Stores vehicle counts
└── README.md


## Approach 1: YOLOv8-Based System

### 1. Description:

This approach uses a pre-trained YOLOv8 model to detect vehicles in four video streams representing four traffic lanes.


### 2. Features:

Real-time vehicle detection
Multi-lane processing using four video inputs
Bounding box visualization
Lane-wise vehicle counting
Adaptive signal timing


### 3. Workflow:

4 video inputs → YOLO detection → vehicle count → out.txt → signal timing calculation
Running the YOLOv8 System


### 4. Install dependencies:

pip install ultralytics opencv-python numpy


### 5. Run the program:

python program.py


### 6. Output:

Four video streams displayed in a single window

Vehicle detection with bounding boxes

Live count of vehicles in each lane

Signal timing printed in the terminal


## Approach 2: Classical Image Processing and Tracking

### 1. Description:

This approach uses background subtraction and centroid tracking to detect and track moving vehicles.


### 2. Techniques Used:

Background subtraction (MOG2)

Thresholding

Morphological operations

Contour detection

Centroid tracking


### 3. Features:

Lightweight and does not require deep learning

Tracks individual vehicles using IDs

Provides vehicle count and density classification

Running the Classical System

python multithreading.py

### 4. Output:

Vehicle detection based on motion

Unique ID tracking for each vehicle

Total vehicle count

Density classification (LOW, MEDIUM, HIGH)


### Signal Timing Logic

Signal timing is calculated proportionally based on vehicle count in each lane.


Formula:

time = (vehicles / total vehicles) × baseTimer


Constraints:

Minimum time: 5 seconds
Maximum time: 30 seconds


### Sample Output:

Input no of vehicles : 7 12 5 6

Signal timings:

Lane 1: 28 sec

Lane 2: 30 sec

Lane 3: 20 sec

Lane 4: 24 sec

Total cycle time: 102 sec


### Future Improvements:

Integration of traffic light simulation

Graphical user interface for monitoring

Advanced vehicle tracking using deep learning (e.g., Deep SORT)

Traffic flow prediction using machine learning


### Conclusion:

This project demonstrates how computer vision techniques and intelligent decision-making can be applied to optimize traffic management systems and reduce congestion.