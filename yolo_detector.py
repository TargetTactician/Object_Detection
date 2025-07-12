from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence=0.3):
        
        if model_path.endswith('.pt') and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        # Perform detection using YOLO with set confidence threshold
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]

        # Format and return the detections
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Get class index and confidence
            class_number = int(box.cls[0])
            conf = float(box.conf[0])

            # Skip low-confidence detections (optional hard threshold)
            if conf < self.confidence:
                continue

            # Append in (bbox, class, confidence) format
            detections.append(([x1, y1, w, h], class_number, conf))

        return detections
