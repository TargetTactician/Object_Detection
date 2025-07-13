# YOLOv11n + Deep SORT Object Detection & Tracking

This project performs real-time object detection and tracking using **YOLOv11n** and **Deep SORT**, either from a webcam or a video file. Detected objects are annotated with bounding boxes and persistent tracking IDs, and the results are saved as output videos.

---

## Demo
The Output Video file is stored in "data/output" directory.

---

## Features

- ✅ Real-time object detection using [Ultralytics YOLOv11n](https://github.com/ultralytics/ultralytics)
- ✅ Object tracking using Deep SORT with appearance-based re-identification
- ✅ Supports webcam and video file inputs
- ✅ Automatically resizes for speed without losing tracking accuracy

---

## Project Structure

```bash
.
├── main.ipynb                  # Entry point for detection and tracking
├── tracker.py              # Deep SORT tracking class
├── yolo_detector.py        # YOLOv8 detection class
├── models/
│   └── yolo11n.pt          # YOLOv8n or custom trained model
├── data/
│   ├── test/people.mp4     # Example input video
│   └── output/             # Output folder for results
└── README.md               # Project documentation
````

---

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/TargetTactician/Mushroom_Classification.git
   cd Mushroom_Classification
   ```

2. **Create a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run the Project

### Webcam Mode

```bash
python main.ipynb
```

### Video File Mode

Inside `main.ipynb`:

```python
VIDEO_PATH = "data/test/people.mp4"  # Use video file instead of webcam
```

---

## Model Notes

* **YOLOv11n** (nano) is used for speed. You can switch to `yolov11s.pt`, `yolov11m.pt`, etc., for better accuracy.
* The model supports 80 COCO classes (person, bottle, cell phone, laptop, etc.)

To list all supported classes:

```python
print(detector.model.names)
```

---

## Applications

* Smart surveillance
* Traffic monitoring
* Industrial automation
* Retail analytics

---

## YOLO + Deep SORT Workflow

1. Detect objects using YOLOv8
2. Extract bounding boxes and classes
3. Feed them to Deep SORT for multi-object tracking
4. Assign consistent track IDs across frames
5. Visualize and save the output

---

## 🙋‍♂️ Author

* **Parthi**
