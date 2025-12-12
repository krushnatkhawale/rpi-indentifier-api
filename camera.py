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
<<<<<<< HEAD
        # Try to load TFLite model if present
        try:
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
        except Exception as e:
            print("Error while loading TFLite model:", e)

        # If no TFLite, setup HOG+SVM fallback for person detection
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("Using HOG fallback detector")
        except Exception as e:
            print("HOG fallback unavailable:", e)
            self.hog = None
=======
        if os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS):
            self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
        else:
            # Setup a fallback HOG + SVM person detector for basic detection when Caffe model is unavailable
            try:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except Exception:
                self.hog = None
>>>>>>> refs/remotes/origin/main

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
<<<<<<< HEAD
                return annotated, detections_count
=======
        else:
            # If Caffe model not available, try HOG person detector as a fallback
            if hasattr(self, 'hog') and self.hog is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w_rec, h_rec), weight in zip(rects, weights):
                    cv2.rectangle(annotated, (x, y), (x + w_rec, y + h_rec), (255, 0, 0), 2)
                    cv2.putText(annotated, f"person: {float(weight):.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detections_count['person'] += 1
            else:
                cv2.putText(annotated, "No detection model loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
>>>>>>> refs/remotes/origin/main

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
