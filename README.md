# Raspberry Pi Video Capture + Object Identifier API

Lightweight Flask app to stream Raspberry Pi camera video to a webpage, run object detection (MobileNetSSD), record video, and save metadata about objects seen during recordings. Also provides a page to view previously recorded feeds.

Features
- Live MJPEG stream to browser with object bounding boxes and labels.
- Start/stop recording from webpage; while recording the server saves video and object counts.
- Recordings listing page with playback and metadata (JSON).

Requirements & notes
- This project uses OpenCV's DNN with MobileNetSSD (Caffe). On Raspberry Pi, prefer installing system OpenCV with video support (libcamera/v4l2). `opencv-python` wheels may not work on Raspbian; use `pip install -r requirements.txt` only if appropriate for your Pi.
- Camera access: the app uses `cv2.VideoCapture(0)` by default. On newer Pi OS with libcamera, ensure a v4l2 loopback or `libcamera-vid` provides `/dev/video0`.

Quick start

1. (Optional) Download model files (prototxt + caffemodel):

```bash
python3 download_model.py
```

2. Install dependencies (on Pi, you may need system packages first):

```bash
python3 -m pip install -r requirements.txt
```

3. Run the app:

```bash
python3 app.py
```

4. Open browser to `http://<pi-ip>:5000/`.

Developer notes
- Model files are expected in `models/` as `MobileNetSSD_deploy.prototxt` and `MobileNetSSD_deploy.caffemodel` (the download script will fetch them to that location).
- Recordings and metadata saved under `recordings/`.