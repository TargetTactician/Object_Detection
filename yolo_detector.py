from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path, confidence):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        # Use YOLO to predict objects in the image with the specified confidence level.
        results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        # Process the detection results and return the formatted detections.
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        # Iterate over each detected object
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Get the class number for the detected object
            class_number = int(box.cls[0])
            # Get the confidence score
            conf = box.conf[0]
            # Add the detection to the list without filtering
            detections.append((([x1, y1, w, h]), class_number, conf))
        return detections
