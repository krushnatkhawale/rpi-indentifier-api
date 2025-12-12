"""Quick camera health check script.

Run this on the Pi to validate that OpenCV can open the camera and read a frame.
"""
import cv2
import sys

def main():
    cap = cv2.VideoCapture(0)
    print("VideoCapture opened:", cap.isOpened())
    if not cap.isOpened():
        print("Unable to open /dev/video0 (or default camera).")
        print("If you're on Raspberry Pi OS with libcamera, either enable v4l2 loopback or use picamera2.")
        sys.exit(1)
    # Try a few reads
    ok = False
    for i in range(5):
        ret, frame = cap.read()
        print(f"read {i}:", ret, "frame shape:", None if frame is None else frame.shape)
        if ret and frame is not None:
            ok = True
            break
    cap.release()
    if not ok:
        print("Camera opened but no frames could be read.")
        sys.exit(2)
    print("Camera check passed.")

if __name__ == '__main__':
    main()
