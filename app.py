from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import os
import time
from camera import camera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    while True:
        jpg = camera.get_frame_jpeg()
        if jpg is None:
            time.sleep(0.1)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    res = camera.start_recording()
    # camera.start_recording now returns a dict with at least 'ok' key
    if isinstance(res, dict):
        return jsonify(res)
    # legacy boolean
    return jsonify({'ok': bool(res)})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    meta = camera.stop_recording()
    if meta is None:
        return jsonify({'ok': False, 'error': 'not recording'})
    return jsonify({'ok': True, 'meta': meta})


@app.route('/recordings')
def recordings():
    rec_dir = 'recordings'
    os.makedirs(rec_dir, exist_ok=True)
    files = []
    for fn in sorted(os.listdir(rec_dir), reverse=True):
        if fn.endswith('.mp4'):
            meta_file = os.path.join(rec_dir, fn + '.json')
            meta = None
            if os.path.exists(meta_file):
                try:
                    import json
                    with open(meta_file) as f:
                        meta = json.load(f)
                except Exception:
                    meta = None
            files.append({'filename': fn, 'meta': meta})
    return render_template('recordings.html', recordings=files)


@app.route('/recordings/<path:filename>')
def recordings_file(filename):
    return send_from_directory('recordings', filename)


@app.route('/status')
def status():
    cap = getattr(camera, 'cap', None)
    cap_opened = False
    cap_width = None
    cap_height = None
    try:
        if cap is not None:
            cap_opened = bool(cap.isOpened())
            try:
                cap_width = cap.get(3)
                cap_height = cap.get(4)
            except Exception:
                cap_width = cap_height = None
    except Exception:
        cap_opened = False

    info = {
        'recording': bool(camera.recording),
        'frame_available': camera.frame is not None,
        'detector': 'tflite' if getattr(camera, 'interpreter', None) is not None else ('hog' if getattr(camera, 'hog', None) is not None else 'none'),
        'cap_opened': cap_opened,
        'cap_width': cap_width,
        'cap_height': cap_height,
    }
    return jsonify(info)


if __name__ == '__main__':
    def _startup_checks():
        print("Startup checks: verifying camera and recorder...")
        cap = getattr(camera, 'cap', None)
        try:
            cap_opened = bool(cap.isOpened()) if cap is not None else False
        except Exception:
            cap_opened = False
        print(f"Camera opened: {cap_opened}")

        # wait for a frame from camera.get_frame_jpeg
        frame_ok = False
        for i in range(50):
            jpg = camera.get_frame_jpeg()
            if jpg:
                frame_ok = True
                break
            time.sleep(0.1)
        print(f"Frame available: {frame_ok}")

        # Test recorder: try to start and immediately stop a short test recording
        test_path = None
        try:
            res = camera.start_recording()
            if isinstance(res, dict):
                if res.get('ok'):
                    test_path = res.get('filename')
                    print(f"Test recording started: {test_path}")
                    # give it a moment to write some frames
                    time.sleep(0.5)
                    meta = camera.stop_recording()
                    if meta:
                        print("Test recording stopped; metadata:", meta)
                        # remove test files
                        try:
                            if test_path and os.path.exists(test_path):
                                os.remove(test_path)
                            metaf = test_path + '.json' if test_path else None
                            if metaf and os.path.exists(metaf):
                                os.remove(metaf)
                        except Exception as e:
                            print("Failed to remove test recording files:", e)
                else:
                    print("Failed to start test recording:", res)
            else:
                # legacy boolean
                if res:
                    print("Test recording started (legacy boolean). Stopping...")
                    meta = camera.stop_recording()
                    print("Test recording metadata:", meta)
                else:
                    print("Failed to start test recording (boolean False)")
        except Exception as e:
            print("Exception during test recording:", e)

    _startup_checks()
    app.run(host='0.0.0.0', port=5000, threaded=True)
