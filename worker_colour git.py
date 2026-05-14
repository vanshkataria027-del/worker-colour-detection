import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def detect_uniform(video_source, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = model(frame)[0]
        
        blue_count = 0
        no_blue_count = 0
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = hsv[y1:y2, x1:x2]
            
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(person_crop, lower_blue, upper_blue)
            
            if mask.sum() > 5000:
                blue_count += 1
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            else:
                no_blue_count += 1
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        
        cv2.putText(frame, f"Blue: {blue_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f"No Blue: {no_blue_count}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

p = argparse.ArgumentParser()
p.add_argument("--video", required=True)
args = p.parse_args()
detect_uniform(args.video)