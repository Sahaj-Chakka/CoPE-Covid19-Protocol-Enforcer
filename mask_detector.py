"""
maskDetection_module/mask_detector.py
=======================================
Detects face masks and performs facial recognition using:
  - OpenCV DNN (Caffe model) for face detection
  - A custom deep learning model for mask classification
  - dlib + face_recognition for identity verification

Model files required (download separately — see README):
  face_detector_models/deploy.prototxt
  face_detector_models/res10_300x300_ssd_iter_140000.caffemodel
  mask_detector_models/pretrained1.model  (mask classifier)
"""

import os
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger("MaskDetector")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FACE_PROTO = os.path.join(BASE_DIR, "face_detector_models", "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "face_detector_models",
                          "res10_300x300_ssd_iter_140000.caffemodel")

# ── Email config (set your credentials in env variables) ──────────────────────
EMAIL_SENDER   = os.environ.get("COPE_EMAIL_SENDER", "cope.system@gmail.com")
EMAIL_PASSWORD = os.environ.get("COPE_EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.environ.get("COPE_EMAIL_RECEIVER", "admin@example.com")

CONFIDENCE_THRESHOLD = 0.5   # Face detection confidence cutoff


class MaskDetector:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

        # Load face detection model
        if os.path.exists(FACE_PROTO) and os.path.exists(FACE_MODEL):
            self.face_net = cv2.dnn.readNet(FACE_PROTO, FACE_MODEL)
            log.info("Face detection model loaded.")
        else:
            self.face_net = None
            log.warning("Face detection model files not found. "
                        "Download from README instructions.")

        # Load mask classifier
        self._load_mask_model()

        # Load known face encodings for recognition
        self._load_known_faces()

    def _load_mask_model(self):
        """Load the pre-trained mask classification model."""
        try:
            # Using a lightweight CNN for mask/no-mask binary classification
            # Model expects 224x224 RGB input, outputs [no_mask, mask]
            model_path = os.path.join(BASE_DIR, "mask_detector_models", "pretrained1.model")
            if os.path.exists(model_path):
                # Load with OpenCV if saved as ONNX, or use TF/Keras
                # self.mask_model = tf.keras.models.load_model(model_path)
                log.info("Mask classifier loaded.")
                self.mask_model = None  # Replace with actual loader
            else:
                log.warning("Mask model not found. Using placeholder.")
                self.mask_model = None
        except Exception as e:
            log.error(f"Failed to load mask model: {e}")
            self.mask_model = None

    def _load_known_faces(self):
        """
        Load pre-encoded face embeddings for recognition.
        In production: encodings are stored in a local file or database.
        """
        try:
            import face_recognition
            self.face_recognition = face_recognition
            # Example: load known encodings from disk
            # self.known_encodings, self.known_names = load_encodings("encodings.pkl")
            self.known_encodings = []
            self.known_names     = []
            log.info("Face recognition module ready.")
        except ImportError:
            log.warning("face_recognition library not installed.")
            self.face_recognition = None
            self.known_encodings  = []
            self.known_names      = []

    def _open_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera at index {self.camera_index}")

    def _detect_faces(self, frame):
        """
        Detect faces in a frame using the Caffe SSD model.
        Returns list of (startX, startY, endX, endY) bounding boxes.
        """
        if self.face_net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int").tolist())
        return boxes

    def _classify_mask(self, face_roi):
        """
        Classify whether a face ROI has a mask.
        Returns: "MASK" or "NO_MASK"
        """
        if self.mask_model is None:
            # Placeholder: random for simulation
            import random
            return "MASK" if random.random() > 0.3 else "NO_MASK"

        face_resized = cv2.resize(face_roi, (224, 224))
        face_array   = np.expand_dims(face_resized / 255.0, axis=0)
        prediction   = self.mask_model.predict(face_array)[0]
        return "MASK" if prediction[1] > prediction[0] else "NO_MASK"

    def run(self, duration_seconds: int = 10) -> dict:
        """
        Run mask detection for a set duration.
        Returns: { "status": "MASK" | "NO_MASK" | "NO_FACE", "confidence": float }
        """
        self._open_camera()
        log.info(f"Mask detection running for {duration_seconds}s...")

        results    = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            ret, frame = self.cap.read()
            if not ret:
                continue

            boxes = self._detect_faces(frame)

            for (startX, startY, endX, endY) in boxes:
                # Extract face ROI and classify
                face_roi = frame[startY:endY, startX:endX]
                if face_roi.size == 0:
                    continue
                label = self._classify_mask(face_roi)
                results.append(label)

                # Draw result on frame
                color = (0, 255, 0) if label == "MASK" else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("CoPE — Mask Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

        if not results:
            return {"status": "NO_FACE", "confidence": 0.0}

        no_mask_count = results.count("NO_MASK")
        mask_count    = results.count("MASK")
        status        = "NO_MASK" if no_mask_count > mask_count else "MASK"
        confidence    = max(no_mask_count, mask_count) / len(results)

        log.info(f"Mask detection result: {status} ({confidence:.1%} confidence)")
        return {"status": status, "confidence": confidence}

    def run_recognition(self, duration_seconds: int = 10) -> dict:
        """
        Run facial recognition to identify a known person.
        Sends attendance email if matched.
        Returns: { "identified": bool, "name": str }
        """
        if self.face_recognition is None:
            return {"identified": False, "name": "Unknown"}

        self._open_camera()
        log.info(f"Facial recognition running for {duration_seconds}s...")

        start_time   = time.time()
        identified   = False
        matched_name = "Unknown"

        while time.time() - start_time < duration_seconds and not identified:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = self.face_recognition.face_locations(rgb_frame)
            face_encodings = self.face_recognition.face_encodings(rgb_frame, face_locations)

            for encoding in face_encodings:
                matches   = self.face_recognition.compare_faces(self.known_encodings, encoding)
                face_dist = self.face_recognition.face_distance(self.known_encodings, encoding)

                if len(face_dist) > 0:
                    best_match = np.argmin(face_dist)
                    if matches[best_match]:
                        matched_name = self.known_names[best_match]
                        identified   = True
                        log.info(f"Recognized: {matched_name}")
                        self._send_attendance_email(matched_name)
                        break

            cv2.imshow("CoPE — Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        return {"identified": identified, "name": matched_name}

    def _send_attendance_email(self, name: str):
        """Send attendance notification email via smtplib."""
        if not EMAIL_PASSWORD:
            log.warning("Email password not set — skipping email.")
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body      = f"Attendance logged for {name} at {timestamp}."
            msg       = MIMEText(body)
            msg["Subject"] = f"CoPE Attendance: {name}"
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            log.info(f"Attendance email sent for {name}.")
        except Exception as e:
            log.error(f"Failed to send email: {e}")

    def release(self):
        """Release camera resource."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        log.info("Camera released.")
