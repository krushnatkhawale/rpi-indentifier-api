"""Download MobileNetSSD model files (prototxt + caffemodel) to ./models.
Run `python3 download_model.py` to fetch files.
"""
import os
import requests

MODEL_URLS = {
    "prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
    ),
    "caffemodel": (
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
    ),
}


def download(url, dest):
    print(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    os.makedirs("models", exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = os.path.join("models", f"MobileNetSSD_deploy.{name}")
        if os.path.exists(dest):
            print(dest, "already exists, skipping")
            continue
        download(url, dest)
    print("Done. Models saved in ./models/")


if __name__ == "__main__":
    main()
