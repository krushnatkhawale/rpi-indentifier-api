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
    success = camera.start_recording()
    return jsonify({'ok': bool(success)})


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
