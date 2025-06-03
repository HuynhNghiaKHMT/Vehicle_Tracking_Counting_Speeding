import cv2
import numpy as np
import time
from configparser import ConfigParser
from detector.detector_RT import Detector_RT
from deepsort_tracker.deep_sort import DeepSort
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator

# === CONFIG SETUP ===
config = ConfigParser()
config.read('tracker.cfg')

# === GLOBAL SETTINGS ===
min_box_area = 100
class_name = ['Motor', 'Car', 'Bus', 'Truck']

def draw_bbox(frame, id, x1, y1, x2, y2, conf, label, speed=None, type='detect'):
    bg_colors = {
        'Motor': (255, 62, 191),
        'Car': (41, 0, 223),
        'Bus': (0, 140, 255),
        'Truck': (0, 215, 255)
    }
    background_color = bg_colors.get(class_name[label], (0, 0, 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15

    if type == "detect":
        text = f"{class_name[label]}:{conf:.2f}"
    else:
        text = f'#{id}-{class_name[label]}'
        if speed is not None:
            text += f':{int(speed)}km/h'

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
    cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)

def main():
    # === LOAD RT-DETR MODEL ===
    weights = config.get('Detector', 'RT_DETR')
    model = Detector_RT(weights)

    # === INIT TRACKER ===
    tracker = DeepSort('ckpt.t7')

    # === LOAD VIDEO ===
    cap = cv2.VideoCapture(config.get('video', 'video_path5'))    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))

    out_tracking = cv2.VideoWriter(config.get('video', 'DS_RT_out_5'), cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (width, height))
    mot_output_path = config.get('annotations', 'anno_pred_5_RT_DS')
    mot_file = open(mot_output_path, 'w')

    frame_id = 0
    tracked_objects = {}
    tracking_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === DETECTION ===
        start_time = time.time()
        detect_results = model.detect(frame)
        detections = []
        current_detections = []
        frame_detected = frame.copy()

        for result in detect_results:
            for box in result.boxes:
                label = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                detections.append([x1, y1, x2, y2, conf, label])
                current_detections.append(((x1, y1, x2, y2), conf, label))
                draw_bbox(frame_detected, 'none', x1, y1, x2, y2, conf, label, type='detect')

        detections = np.array(detections) if detections else np.empty((0, 6))

        # === TRACKING ===
        online_targets = tracker.update(detections, [height, width], [height, width], frame)
        frame_tracked = frame.copy()

        for t in online_targets:
            x1, y1, x2, y2, tid, class_id = map(int, t)
            w, h = x2 - x1, y2 - y1

            if w * h < min_box_area:
                continue

            tracked_objects[tid] = class_id

            tracking_results.append(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")
            mot_file.write(f"{frame_id+1},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")

            draw_bbox(frame_tracked, tid, x1, y1, x2, y2, 0.0, class_id, type='track')

        # === OUTPUT ===
        out_tracking.write(frame_tracked)
        frame_id += 1
        print(f"[INFO] Frame {frame_id}, FPS: {1 / (time.time() - start_time):.2f}")

    # === CLEANUP ===
    cap.release()
    mot_file.close()
    out_tracking.release()
    print(f"✅ Tracking results saved to: {mot_output_path}")

if __name__ == "__main__":
    main()
