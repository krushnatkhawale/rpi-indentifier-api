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
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def main():
    os.makedirs("models", exist_ok=True)
    downloaded_any = False
    for name, url_list in MODEL_URLS.items():
        dest = os.path.join("models", f"MobileNetSSD_deploy.{name}")
        if os.path.exists(dest):
            print(dest, "already exists, skipping")
            downloaded_any = True
            continue

        last_err = None
        for url in url_list:
            try:
                # quick HEAD to check availability when possible
                try:
                    head = requests.head(url, allow_redirects=True, timeout=10)
                    if head.status_code != 200:
                        print(f"URL not available (status={head.status_code}): {url}")
                        continue
                except Exception:
                    # some hosts don't respond to HEAD; fallback to GET trial
                    pass

                download(url, dest)
                downloaded_any = True
                break
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                last_err = e
        if not os.path.exists(dest):
            print(f"Could not obtain MobileNetSSD_deploy.{name}.\nPlease download a compatible model/prototxt manually and place it at: {dest}")
            if last_err:
                print("Last error:", last_err)

    if downloaded_any:
        print("Done. Models saved in ./models/")
    else:
        print("No files were downloaded. See the messages above.\nIf the original MobileNet-SSD repository is unavailable, you can manually obtain compatible files by searching for 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel' and placing them into the './models' folder.")


if __name__ == "__main__":
    main()
