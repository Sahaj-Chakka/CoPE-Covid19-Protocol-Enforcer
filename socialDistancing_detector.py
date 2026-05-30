"""
socialDistancing_module/socialDistancing_detector.py
======================================================
Detects social distancing violations using YOLOv3 + OpenCV.

Pipeline:
  1. Capture frame from webcam
  2. Run YOLOv3 to detect humans (class 0 in COCO)
  3. Get bounding boxes for each person
  4. Check if any boxes intersect or centers are too close
  5. Flag violations and draw RED boxes around violators

Model files required (download separately — see README):
  yolov3/yolov3.cfg
  yolov3/yolov3.weights   (~237MB, download from pjreddie.com)
  yolov3/coco.names

YOLOv3 ARM-optimized weights can be used for better Pi performance.
"""

import os
import time
import logging
import cv2
import numpy as np

log = logging.getLogger("SocialDistancing")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
YOLO_CFG      = os.path.join(BASE_DIR, "yolov3", "yolov3.cfg")
YOLO_WEIGHTS  = os.path.join(BASE_DIR, "yolov3", "yolov3.weights")
YOLO_NAMES    = os.path.join(BASE_DIR, "yolov3", "coco.names")

# Detection config
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD        = 0.4
PERSON_CLASS_ID      = 0     # In COCO dataset, 'person' is class 0
MIN_DISTANCE_PX      = 100   # Minimum safe distance between bounding box centers (pixels)


class SocialDistancingDetector:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap          = None
        self.net          = None
        self.layer_names  = None

        self._load_yolo()

    def _load_yolo(self):
        """Load YOLOv3 model from disk."""
        if os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS):
            self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
            # Use CPU (Raspberry Pi doesn't have CUDA)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            layer_names       = self.net.getLayerNames()
            out_layers        = self.net.getUnconnectedOutLayers()
            self.layer_names  = [layer_names[i - 1] for i in out_layers.flatten()]

            log.info("YOLOv3 model loaded successfully.")
        else:
            log.warning("YOLOv3 model files not found. "
                        "Download from: https://pjreddie.com/media/files/yolov3.weights")
            self.net = None

        # Load COCO class labels
        if os.path.exists(YOLO_NAMES):
            with open(YOLO_NAMES, "r") as f:
                self.classes = f.read().strip().split("\n")
        else:
            self.classes = []

    def _detect_people(self, frame) -> list:
        """
        Run YOLO inference on a frame.
        Returns list of bounding boxes: [(x, y, w, h), ...]
        where (x, y) = top-left corner, (w, h) = width/height
        """
        if self.net is None:
            return []

        h, w = frame.shape[:2]
        blob  = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                       swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        boxes, confidences = [], []

        for output in outputs:
            for detection in output:
                scores     = detection[5:]
                class_id   = np.argmax(scores)
                confidence = scores[class_id]

                # Only keep person detections above threshold
                if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w    = int(detection[2] * w)
                    box_h    = int(detection[3] * h)
                    x        = center_x - box_w // 2
                    y        = center_y - box_h // 2

                    boxes.append([x, y, box_w, box_h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                    CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(indices) == 0:
            return []

        return [boxes[i] for i in indices.flatten()]

    def _get_center(self, box: list) -> tuple:
        """Return center (cx, cy) of a bounding box."""
        x, y, w, h = box
        return (x + w // 2, y + h // 2)

    def _boxes_intersect(self, box1: list, box2: list) -> bool:
        """Check if two bounding boxes overlap."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Get all corners
        left1,  right1  = x1, x1 + w1
        top1,   bottom1 = y1, y1 + h1
        left2,  right2  = x2, x2 + w2
        top2,   bottom2 = y2, y2 + h2

        # Check overlap condition
        return not (right1 < left2 or right2 < left1 or
                    bottom1 < top2 or bottom2 < top1)

    def _euclidean_distance(self, center1: tuple, center2: tuple) -> float:
        """Calculate Euclidean distance between two center points."""
        return np.sqrt((center1[0] - center2[0])**2 +
                       (center1[1] - center2[1])**2)

    def _check_violations(self, boxes: list) -> list:
        """
        Identifies pairs of people violating social distance.
        Returns list of box indices that are in violation.
        """
        violating_indices = set()

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                center_i = self._get_center(boxes[i])
                center_j = self._get_center(boxes[j])

                dist      = self._euclidean_distance(center_i, center_j)
                intersect = self._boxes_intersect(boxes[i], boxes[j])

                if dist < MIN_DISTANCE_PX or intersect:
                    violating_indices.add(i)
                    violating_indices.add(j)
                    log.debug(f"Violation: person {i} and {j}, dist={dist:.0f}px")

        return list(violating_indices)

    def _draw_results(self, frame, boxes, violating_indices):
        """Draw bounding boxes: GREEN for safe, RED for violation."""
        for i, (x, y, w, h) in enumerate(boxes):
            color  = (0, 0, 255) if i in violating_indices else (0, 255, 0)
            label  = "VIOLATION" if i in violating_indices else "SAFE"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show count summary
        total     = len(boxes)
        violating = len(violating_indices)
        summary   = f"People: {total} | Violations: {violating}"
        cv2.putText(frame, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return frame

    def run(self, duration_seconds: int = 10) -> dict:
        """
        Run social distancing detection for a set duration.
        Returns: { "violation": bool, "people_count": int, "violations_count": int }
        """
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        log.info(f"Social distancing detection running for {duration_seconds}s...")

        violation_detected = False
        max_people         = 0
        max_violations     = 0
        start_time         = time.time()

        while time.time() - start_time < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                continue

            boxes             = self._detect_people(frame)
            violating_indices = self._check_violations(boxes)

            max_people     = max(max_people, len(boxes))
            max_violations = max(max_violations, len(violating_indices))

            if violating_indices:
                violation_detected = True

            frame = self._draw_results(frame, boxes, violating_indices)
            cv2.imshow("CoPE — Social Distancing", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

        result = {
            "violation":        violation_detected,
            "people_count":     max_people,
            "violations_count": max_violations
        }
        log.info(f"Social distancing result: {result}")
        return result

    def release(self):
        """Release camera resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
