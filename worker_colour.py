import cv2
import numpy as np
import argparse
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

BLUE_HSV_LOWER = np.array([90, 60, 40], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([130, 255, 255], dtype=np.uint8)
BLUE_FRACTION_THRESHOLD = 0.08

def blue_fraction(frame_bgr, box):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    total_pixels = (x2 - x1) * (y2 - y1)
    blue_pixels = int(np.sum(mask > 0))
    return blue_pixels / total_pixels if total_pixels > 0 else 0.0

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for r in results.boxes:
            if int(r.cls[0]) == 0 and float(r.conf[0]) > 0.4:
                boxes.append(r.xyxy[0].cpu().numpy())
        return boxes

class HOGDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    def detect(self, frame):
        resized = cv2.resize(frame, (640, 480))
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 480
        rects, _ = self.hog.detectMultiScale(resized, winStride=(8,8), padding=(4,4), scale=1.05)
        boxes = []
        for (x, y, w, h) in rects:
            boxes.append(np.array([int(x*scale_x), int(y*scale_y), int((x+w)*scale_x), int((y+h)*scale_y)]))
        return boxes

def process_video(source, output_path=None, model_name="yolov8n.pt", show=True):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    detector = YOLODetector(model_name) if YOLO_AVAILABLE else HOGDetector()
    frame_idx = 0
    skip_frames = max(1, int(fps // 5))
    max_blue = max_no_blue = 0
    count_blue = count_no_blue = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % skip_frames == 0:
            boxes = detector.detect(frame)
            count_blue = count_no_blue = 0
            for box in boxes:
                frac = blue_fraction(frame, box)
                has_blue = frac >= BLUE_FRACTION_THRESHOLD
                if has_blue:
                    count_blue += 1
                    colour, label = (200, 100, 0), f"Blue Uniform ({frac*100:.0f}%)"
                else:
                    count_no_blue += 1
                    colour, label = (0, 0, 220), "No Uniform"
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), colour, -1)
                cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            max_blue    = max(max_blue,    count_blue)
            max_no_blue = max(max_no_blue, count_no_blue)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Blue Uniform : {count_blue}",  (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,200,255), 2)
        cv2.putText(frame, f"No Uniform   : {count_no_blue}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50,50,255), 2)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Uniform Detector [Q=quit]", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"\nBlue Uniform (peak): {max_blue}")
    print(f"No Uniform   (peak): {max_no_blue}")

p = argparse.ArgumentParser()
src = p.add_mutually_exclusive_group(required=True)
src.add_argument("--video",  type=str)
src.add_argument("--camera", type=int)
p.add_argument("--output", type=str, default=None)
p.add_argument("--model",  type=str, default="yolov8n.pt")
p.add_argument("--no-display", action="store_true")
args = p.parse_args()
source = args.video if args.video else args.camera
process_video(source, args.output, args.model, not args.no_display)