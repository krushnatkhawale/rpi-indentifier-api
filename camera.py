"""Camera capture + object detection using TFLite (fallback to HOG person detector).

Place a TF Lite detection model at `models/model.tflite` (or use `download_model.py --url ...`).
"""
import threading
import time
import os
import json
from collections import defaultdict

import cv2
import numpy as np

# Try to import the lightweight tflite runtime first (recommended on Raspberry Pi).
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter
    except Exception:
        Interpreter = None

# Path to expected TFLite model
TFLITE_MODEL = os.path.join("models", "model.tflite")

# COCO labels (index 0 unused).
COCO_LABELS = [
    "", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack", "umbrella",
    "", "", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "", "dining table",
    "", "", "toilet", "", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
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

        # inference related
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.conf_threshold = 0.4

        # Initialize detector and start capture thread
        self._ensure_model()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _ensure_model(self):
        """Try to load a TFLite model. If not available, prepare HOG fallback."""
        if Interpreter is not None and os.path.exists(TFLITE_MODEL):
            try:
                self.interpreter = Interpreter(model_path=TFLITE_MODEL)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("Loaded TFLite model:", TFLITE_MODEL)
                return
            except Exception as e:
                print("Failed to initialize TFLite interpreter:", e)
                self.interpreter = None

        # HOG fallback
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("Using HOG fallback detector")
        except Exception as e:
            print("HOG fallback unavailable:", e)
            self.hog = None

    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                annotated, detections = self._process_frame(frame)
                with self.lock:
                    self.frame = annotated
                    if self.recording and self.record_writer is not None:
                        try:
                            self.record_writer.write(annotated)
                            self.frame_count += 1
                            for cls, count in detections.items():
                                self.record_meta["object_counts"][cls] += count
                        except Exception as e:
                            print("Error while writing frame to recorder:", e)
                time.sleep(0.005)
            except Exception as e:
                print("Unexpected error in capture loop:", e)
                time.sleep(0.5)

    def _run_tflite(self, frame):
        """Run inference and return list of (label, score, (x1,y1,x2,y2))."""
        results = []
        try:
            if not self.interpreter or not self.input_details:
                return results

            inp = self.input_details[0]
            # input shape can be (1,H,W,3)
            shape = inp.get('shape')
            if shape is None or len(shape) < 3:
                return results
            h_in, w_in = int(shape[1]), int(shape[2])
            dtype = inp.get('dtype')

            resized = cv2.resize(frame, (w_in, h_in))
            if dtype == np.uint8:
                input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            else:
                input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Collect outputs
            out_tensors = [self.interpreter.get_tensor(o['index']) for o in self.output_details]

            # Heuristics: typical detection models return [boxes, classes, scores, num]
            boxes, classes, scores = None, None, None
            for t in out_tensors:
                if t.ndim == 3 and t.shape[2] == 4:
                    boxes = t
                elif t.ndim == 2 and (t.dtype == np.float32 or t.dtype == np.float64):
                    # could be scores or classes (floats)
                    if scores is None:
                        scores = t
                    else:
                        classes = t
                elif t.ndim == 2 and t.dtype == np.int32:
                    classes = t

            # Try other fallback matching by shape
            if boxes is None:
                for t in out_tensors:
                    if t.ndim == 3 and t.shape[2] == 4:
                        boxes = t
                        break
            if scores is None:
                for t in out_tensors:
                    if t.ndim == 2 and (t.dtype == np.float32 or t.dtype == np.float64):
                        scores = t
                        break
            if classes is None:
                for t in out_tensors:
                    if t.ndim == 2 and (t.dtype == np.float32 or t.dtype == np.int32):
                        classes = t
                        break

            if boxes is None or scores is None or classes is None:
                return results

            # Determine number of detections
            num = scores.shape[1]
            h, w = frame.shape[:2]
            for i in range(num):
                score = float(scores[0, i])
                if score < self.conf_threshold:
                    continue
                cls_id = int(classes[0, i])
                label = COCO_LABELS[cls_id] if 0 <= cls_id < len(COCO_LABELS) else f"id:{cls_id}"
                box = boxes[0, i]
                # Expect normalized [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = box
                x1 = max(0, int(xmin * w))
                y1 = max(0, int(ymin * h))
                x2 = min(w - 1, int(xmax * w))
                y2 = min(h - 1, int(ymax * h))
                results.append((label, score, (x1, y1, x2, y2)))

        except Exception as e:
            print("TFLite inference failed:", e)
        return results

    def _process_frame(self, frame):
        detections_count = defaultdict(int)
        annotated = frame.copy()
        try:
            # Primary: TFLite model
            if self.interpreter is not None:
                dets = self._run_tflite(frame)
                for label, score, (x1, y1, x2, y2) in dets:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label}: {score:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detections_count[label] += 1
                return annotated, detections_count

            # Fallback: HOG person detector
            if hasattr(self, 'hog') and self.hog is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w_rec, h_rec), weight in zip(rects, weights):
                    cv2.rectangle(annotated, (x, y), (x + w_rec, y + h_rec), (255, 0, 0), 2)
                    cv2.putText(annotated, f"person: {float(weight):.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detections_count['person'] += 1
                return annotated, detections_count

            # No detector available
            cv2.putText(annotated, "No detection model available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print("Error processing frame:", e)
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
            if not writer.isOpened():
                print("Failed to open video writer for:", out_path)
                return False
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
                try:
                    self.record_writer.release()
                except Exception as e:
                    print("Error releasing writer:", e)
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
            try:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print("Failed to write metadata:", e)
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
            try:
                self.cap.release()
            except Exception:
                pass


camera = Camera()
import threading
import time
import os
import json
from collections import defaultdict

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
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.conf_threshold = 0.4
        self._ensure_model()
        self.thread.start()

    def _ensure_model(self):
        if os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS):
            self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
        else:
            # Setup a fallback HOG + SVM person detector for basic detection when Caffe model is unavailable
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None

    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                annotated, detections = self._process_frame(frame)
                with self.lock:
                    self.frame = annotated
                    if self.recording and self.record_writer is not None:
                        try:
                            self.record_writer.write(annotated)
                            self.frame_count += 1
                            for cls, count in detections.items():
                                self.record_meta["object_counts"][cls] += count
                        except Exception as e:
                            print("Error while writing frame to recorder:", e)
                time.sleep(0.01)
            except Exception as e:
                print("Unexpected error in capture loop:", e)
                time.sleep(0.5)

    def _run_tflite(self, frame):
        # Run inference with TFLite interpreter, return detections list of (label, score, (x1,y1,x2,y2))
        detections = []
        try:
            inp = self.input_details[0]
            h_in, w_in = inp['shape'][1], inp['shape'][2]
            input_dtype = inp['dtype']

            resized = cv2.resize(frame, (w_in, h_in))
            if input_dtype == np.uint8:
                input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            else:
                # float input
                input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Typical TFLite SSD outputs: boxes, classes, scores, num
            out_tensors = {od['name']: self.interpreter.get_tensor(od['index']) for od in self.output_details}

            # Try to find outputs by common suffixes/names
            boxes = None; classes = None; scores = None; num = None
            for name, val in out_tensors.items():
                lname = name.lower()
                if 'box' in lname or 'location' in lname:
                    boxes = val
                elif 'class' in lname:
                    classes = val
                elif 'score' in lname or 'prob' in lname:
                    scores = val
                elif 'num' in lname:
                    num = val

            # Fallback by shape if name matching failed
            if boxes is None:
                for val in out_tensors.values():
                    if val.ndim == 3 and val.shape[2] == 4:
                        boxes = val
                        break
            if scores is None:
                for val in out_tensors.values():
                    if val.ndim == 2 and val.shape[1] >= 1:
                        # heuristics: scores often shape (1, num)
                        if val.dtype == np.float32 or val.dtype == np.float64:
                            scores = val
                            break
            if classes is None:
                for val in out_tensors.values():
                    if val.dtype == np.float32 or val.dtype == np.float64 or val.dtype == np.int32:
                        if val.ndim == 2 and val.shape[1] >= 1:
                            classes = val
                            break

            if boxes is None or scores is None or classes is None:
                return []

            num_detections = int(num.flatten()[0]) if (num is not None) else scores.shape[1]

            h, w = frame.shape[:2]
            for i in range(num_detections):
                score = float(scores[0, i])
                if score < self.conf_threshold:
                    continue
                cls_id = int(classes[0, i])
                # many models use 1-based class ids; ensure we map correctly
                label = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else f"id:{cls_id}"
                # boxes may be [ymin, xmin, ymax, xmax] normalized
                box = boxes[0, i]
                if box[0] >= 0 and box[2] >= 0:
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * w); y1 = int(ymin * h); x2 = int(xmax * w); y2 = int(ymax * h)
                else:
                    # unexpected format; skip
                    continue
                detections.append((label, score, (x1, y1, x2, y2)))
        except Exception as e:
            print("TFLite inference failed:", e)
        return detections

    def _process_frame(self, frame):
        detections_count = defaultdict(int)
        annotated = frame.copy()
        try:
            if self.interpreter is not None:
                dets = self._run_tflite(frame)
                for label, score, (x1, y1, x2, y2) in dets:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{label}: {score:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detections_count[label] += 1
                return annotated, detections_count

            # Fallback: HOG person detector
            if hasattr(self, 'hog') and self.hog is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w_rec, h_rec), weight in zip(rects, weights):
                    cv2.rectangle(annotated, (x, y), (x + w_rec, y + h_rec), (255, 0, 0), 2)
                    cv2.putText(annotated, f"person: {float(weight):.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detections_count['person'] += 1
                return annotated, detections_count

            cv2.putText(annotated, "No detection model available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print("Error processing frame:", e)
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
                try:
                    self.record_writer.release()
                except Exception as e:
                    print("Error releasing writer:", e)
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
            try:
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print("Failed to write metadata:", e)
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
            try:
                self.cap.release()
            except Exception:
                pass


camera = Camera()
