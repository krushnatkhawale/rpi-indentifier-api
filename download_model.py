"""Download MobileNetSSD model files (prototxt + caffemodel) to ./models.
Run `python3 download_model.py` to fetch files.
"""
import os
import requests

# Candidate URLs for the MobileNetSSD Caffe model and prototxt.
# Historically these lived in chuanqi305/MobileNet-SSD, but some mirrors have been removed.
# The script will try each URL in order and skip ones that return non-200 responses.
MODEL_URLS = {
    "prototxt": [
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/ssd_mobilenet_v1_coco.pbtxt",
    ],
    "caffemodel": [
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
        # fallback candidates (may not exist); keep as references for manual download
    ],
}


def download(url, dest):
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    """Helper to download a TensorFlow Lite object detection model (.tflite) to ./models.

    Usage:
      python3 download_model.py --url <MODEL_URL> [--dest models/model.tflite]

    If no URL is provided the script prints recommended official sources and instructions.
    """
    import argparse
    import os
    import sys
    import requests


    def download(url, dest):
        print(f"Downloading {url} -> {dest}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--url", help="Direct URL to a .tflite model file")
        parser.add_argument("--dest", default="models/model.tflite", help="Destination path")
        args = parser.parse_args()

        if not args.url:
            print("No URL provided. This helper downloads a .tflite model to ./models.")
            print("Official TF Lite object detection models and instructions:")
            print("  - https://www.tensorflow.org/lite/models/object_detection/overview")
            print("")
            print("Common workflow:")
            print("  1. On your development machine, download a TF Lite detection model (for example SSD MobileNet v2 320x320).")
            print("  2. Copy the .tflite file to this project's ./models/ directory as 'model.tflite'.")
            print("")
            print("If you do have a direct URL, run:")
            print("  python3 download_model.py --url https://.../model.tflite --dest models/model.tflite")
            sys.exit(0)

        os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
        try:
            download(args.url, args.dest)
            print("Download complete ->", args.dest)
        except Exception as e:
            print("Failed to download model:", e)
            print("You can manually download a TF Lite detection model from the TensorFlow site and place it at:")
            print("  ./models/model.tflite")


    if __name__ == "__main__":
        main()
