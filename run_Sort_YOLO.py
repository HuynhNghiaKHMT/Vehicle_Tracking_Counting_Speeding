import cv2
import numpy as np
import time
from configparser import ConfigParser
from detector.detector import Detector
from sort_tracker.sort import Sort


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
    # === LOAD YOLOv8 MODEL ===
    model = Detector("model/YOLOv8m.pt")

    # === INIT TRACKER ===
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # === LOAD VIDEO ===
    cap = cv2.VideoCapture(config.get('video', 'video_path5'))    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))

    out_tracking = cv2.VideoWriter(config.get('video', 'S_YOLO_out_5'), cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (width, height))
    mot_output_path = config.get('annotations', 'anno_pred_5_YOLO_S')
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
            for boxes in result.boxes:
                label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)

                #*detections bbox format for SORT
                detections.append([x1, y1, x2, y2, conf])
                current_detections.append(((x1, y1, x2, y2), conf, class_id))
                draw_bbox(frame_detected, 'none', x1, y1, x2, y2, conf, class_id, type='detect')

        detections = np.array(detections) if detections else np.empty((0, 5))

        # === TRACKING ===
        online_targets = tracker.update(detections)
        frame_tracked = frame.copy()

        for t in online_targets:
            # print(t)
            x1, y1, x2, y2, tid = map(int, t)# hoặc một giá trị mặc định
            w, h = x2 - x1, y2 - y1

            matched_label = -1
            min_distance = float('inf')
            center_tracked_x = (x1 + x2) / 2
            center_tracked_y = (y1 + y2) / 2

            for det_bbox, det_conf, det_label in current_detections:
                center_det_x = (det_bbox[0] + det_bbox[2]) / 2
                center_det_y = (det_bbox[1] + det_bbox[3]) / 2
                distance = (center_tracked_x - center_det_x)**2 + (center_tracked_y - center_det_y)**2
                if distance < min_distance:
                    min_distance = distance
                    matched_label = det_label

            tracked_objects[tid] = matched_label

            if w * h > min_box_area:
                tracking_results.append(
                    f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n"
                )
                mot_file.write(f"{frame_id+1},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1\n")
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, 1.0, tracked_objects.get(tid, -1), type='track')
            

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
