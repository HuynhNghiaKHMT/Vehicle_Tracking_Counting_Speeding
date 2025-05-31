from configparser import ConfigParser
from detector.detector import Detector
from sort_tracker.sort import Sort
import cv2
import numpy as np
import time

# SETUP config
config = ConfigParser()
config.read('tracker.cfg')

# SETUP video
cap = cv2.VideoCapture(config.get('video', 'video_path1'))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_tracking = cv2.VideoWriter(config.get('video', 'video_out_tracking'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(config.get('video', 'video_out_detect'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# extra params
aspect_ratio_thresh = 0.6 # more condition for vertical box if you like
min_box_area = 100 # minimum area of the tracking box to be considered
class_name = ['Motor', 'Car', 'Bus', 'Truck']

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def draw_fps(frame, fps):
    fps_text = f' FPS: {fps:.4f}' + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0) # Màu đen
    background_color = (255, 255, 255) # Màu trắng
    text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]
    text_x, text_y = 40, 50
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), background_color, -1)
    cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, label, type='detect'):
    # Định nghĩa màu sắc cho từng lớp
    bg_colors = {
        'Motor': (255, 62, 191),   # Tím tươi
        'Car':   (41, 0, 223),     # Đỏ tươi
        'Bus':   (0, 140, 255),     # Cam tươi
        'Truck': (0, 215, 255)      # vàng tươi
    }

    # Lấy màu sắc tương ứng với nhãn, mặc định là màu đen nếu không tìm thấy
    background_color = bg_colors.get(class_name[label], (0, 0, 0))

    # Vẽ bounding box với màu viền tương ứng
    cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)   # Màu chữ luôn là đen
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15   # Đảm bảo chữ không bị ra ngoài khung hình

    if type == "detect":
        text = f"{class_name[label]}:{conf:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)
    elif type == "track":
        text = f'#{id}:{class_name[label]}'
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)

# For future these functions above can be moved to a separate file
#================================================================================================
def main():
    weights = "best.pt"
    model = Detector(weights)
    args = TrackerArgs(
        track_thresh=config.getfloat('Tracker', 'track_thresh'),
        track_buffer=config.getint('Tracker', 'track_buffer'),
        match_thresh=config.getfloat('Tracker', 'match_thresh'),
        fuse_score=config.getboolean('Tracker', 'fuse_score')
    )
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3) # Initialize SORT tracker
    frame_id = 0
    tracking_results = [] # store tracking results for eval, debug,...
    tracked_objects = {} # Dictionary to store tracked objects with their labels

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #! Detect
        start_time = time.time()
        detect_results = model.detect(frame)
        detections = []
        frame_detected = frame.copy()
        current_detections = [] # Store detections with label

        for result in detect_results:
            for boxes in result.boxes:
                label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]

                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)

                #*detections bbox format for SORT
                detections.append([x1, y1, x2, y2, conf])
                current_detections.append(((x1, y1, x2, y2), conf, class_id)) # Store bbox, conf, and label

                draw_bbox(frame_detected, 'none', x1, y1, x2, y2, conf, class_id, type='detect') # Sử dụng class_id làm label cho detect

        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        
        #! Update tracker with detections format
        online_targets = tracker.update(detections) # SORT expects a list of detections
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # Draw tracked objects
        frame_tracked = frame.copy()
        for t in online_targets:
            x1, y1, x2, y2, tid = map(int, t)
            w, h = x2 - x1, y2 - y1

            # Find the label of the tracked object based on the initial detection
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
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, 1.0, tracked_objects.get(tid, -1), type='track')


        # Write and display the frame
        draw_fps(frame_detected, fps)
        draw_fps(frame_tracked, fps)
        out_detect.write(frame_detected)
        out_tracking.write(frame_tracked)

        frame_id += 1
        # cv2.imshow('frame', frame_tracked)
        # cv2.imshow('frame', frame_detected)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out_detect.release()
    out_tracking.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()