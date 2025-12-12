import threading
import time
import os
import json
from collections import defaultdict

import cv2
import numpy as np

MODEL_PROTO = os.path.join("models", "MobileNetSSD_deploy.prototxt")
MODEL_WEIGHTS = os.path.join("models", "MobileNetSSD_deploy.caffemodel")

CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class Camera:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.recording = False
        self.record_writer = None
        self.record_start = None
        self.record_meta = None
        self.frame_count = 0
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.net = None
        self.conf_threshold = 0.5
        self._ensure_model()
        self.thread.start()

    def _ensure_model(self):
        if os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS):
            self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            annotated, detections = self._process_frame(frame)
            with self.lock:
                self.frame = annotated
                if self.recording and self.record_writer is not None:
                    self.record_writer.write(annotated)
                    self.frame_count += 1
                    # accumulate counts
                    for cls, count in detections.items():
                        self.record_meta["object_counts"][cls] += count
            time.sleep(0.01)

    def _process_frame(self, frame):
        detections_count = defaultdict(int)
        annotated = frame.copy()
        (h, w) = frame.shape[:2]
        if self.net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.conf_threshold:
                    idx = int(detections[0, 0, i, 1])
                    if idx >= len(CLASS_NAMES):
                        continue
                    label = CLASS_NAMES[idx]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(annotated, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(annotated, f"{label}: {confidence:.2f}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detections_count[label] += 1
        else:
            # No model: annotate that detection is disabled
            cv2.putText(annotated, "No detection model loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated, detections_count

    def get_frame_jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            if not ret:
                return None
            return jpeg.tobytes()

    def start_recording(self, out_path=None, fps=20):
        with self.lock:
            if self.recording:
                return False
            os.makedirs('recordings', exist_ok=True)
            if out_path is None:
                ts = time.strftime('%Y%m%d-%H%M%S')
                out_path = os.path.join('recordings', f'recording-{ts}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # wait for a frame to get size
            start = time.time()
            while self.frame is None and time.time() - start < 5:
                time.sleep(0.1)
            if self.frame is None:
                return False
            h, w = self.frame.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            self.recording = True
            self.record_writer = writer
            self.record_start = time.time()
            self.frame_count = 0
            self.record_meta = {"start_time": self.record_start, "object_counts": defaultdict(int), "frames": 0, "filename": out_path}
            return True

    def stop_recording(self):
        with self.lock:
            if not self.recording:
                return None
            self.recording = False
            if self.record_writer is not None:
                self.record_writer.release()
            end_time = time.time()
            duration = end_time - self.record_start if self.record_start else 0
            meta = {
                "start_time": self.record_meta.get("start_time"),
                "end_time": end_time,
                "duration": duration,
                "frames": self.frame_count,
                "object_counts": dict(self.record_meta.get("object_counts", {})),
                "filename": self.record_meta.get("filename"),
            }
            # write metadata JSON next to file
            meta_path = meta["filename"] + ".json"
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            # clear writer and meta
            self.record_writer = None
            self.record_meta = None
            self.record_start = None
            self.frame_count = 0
            return meta

    def shutdown(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()


camera = Camera()
