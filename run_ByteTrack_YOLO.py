
import cv2
import numpy as np
import time
from configparser import ConfigParser
from detector.detector import Detector
from detector.detector_RT import Detector_RT
from bytetrack_tracker.byte_tracker import BYTETracker
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator
from collections import defaultdict
import xml.etree.ElementTree as ET

# SETUP config
config = ConfigParser()
config.read('tracker.cfg')

# SETUP video


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

# def draw_fps(frame, fps):
#     fps_text = f' FPS: {fps:.4f}' + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_color = (0, 0, 0) # Màu đen
#     background_color = (255, 255, 255) # Màu trắng
#     text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]
#     text_x, text_y = 40, 50
#     cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), background_color, -1)
#     cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
#     return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, label,speed=None, type='detect'):
    # Định nghĩa màu sắc cho từng lớp
    bg_colors = {
        'Motor': (255, 62, 191),   # Tím tươi
        'Car':   (41, 0, 223),      # Đỏ tươi
        'Bus':   (0, 140, 255),      # Cam tươi
        'Truck': (0, 215, 255)     # vàng tươi
    }

    # Lấy màu sắc tương ứng với nhãn, mặc định là màu đen nếu không tìm thấy
    background_color = bg_colors.get(class_name[label], (0, 0, 0))

    # Vẽ bounding box với màu viền tương ứng
    cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # Màu chữ luôn là đen
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15  # Đảm bảo chữ không bị ra ngoài khung hình

    if type == "detect":
        text = f"{class_name[label]}:{conf:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)
    elif type == "track":  
        text = f'#{id}-{class_name[label]}'
        #add speed
        if speed is not None:
            text += f':{int(speed)}km/h'

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)

def main():
    # YOLOv8
    weights = "weights\YOLOv8.pt"
    model = Detector(weights)
   
    args = TrackerArgs(
        track_thresh=config.getfloat('Tracker', 'track_thresh'),
        track_buffer=config.getint('Tracker', 'track_buffer'),
        match_thresh=config.getfloat('Tracker', 'match_thresh'),
        fuse_score=config.getboolean('Tracker', 'fuse_score')
    )
    tracker = BYTETracker(args)
    frame_id = 0
    tracking_results = [] # store tracking results for eval, debug,...
    tracked_objects = {} # Dictionary to store tracked objects with their labels

    # SETUP output video
    cap = cv2.VideoCapture(config.get('video', 'video_path2'))    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_tracking = cv2.VideoWriter(config.get('video', 'YOLO_out_2'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    # out_detect = cv2.VideoWriter(config.get('video', f'video_out_detect{i}'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
    mot_output_path = config.get('annotations', 'anno_pred_2_YOLO_BT')
    
    mot_file = open(mot_output_path, 'w')

    target_width = 32
    target_height = 140
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    i = 0
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

                # Detections bbox format for BYTETracker
                if conf > 0.5:
                    detections.append([x1, y1, x2, y2, conf])
                    current_detections.append(((x1, y1, x2, y2), conf, class_id)) # Store bbox, conf, and label

                    draw_bbox(frame_detected, None, x1, y1, x2, y2, conf, class_id, None, type='detect') # Sử dụng class_id làm label cho detect

        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        #! Update tracker with detections format
        online_targets = tracker.update(detections, [height, width], [height, width]) #img_info and img_size is for scaling img, if not then just pass [height, width]
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Draw tracked objects
        frame_tracked = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            t_bbox = tuple(map(int, (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])))

            # Find the label of the tracked object based on the initial detection
            matched_label = -1
            min_distance = float('inf')
            center_tracked_x = (t_bbox[0] + t_bbox[2]) / 2
            center_tracked_y = (t_bbox[1] + t_bbox[3]) / 2

            for det_bbox, det_conf, det_label in current_detections:
                center_det_x = (det_bbox[0] + det_bbox[2]) / 2
                center_det_y = (det_bbox[1] + det_bbox[3]) / 2
                distance = (center_tracked_x - center_det_x)**2 + (center_tracked_y - center_det_y)**2
                if distance < min_distance:
                    min_distance = distance
                    matched_label = det_label

            # Store the label for the tracked ID
            tracked_objects[tid] = matched_label

            if tlwh[2] * tlwh[3] > min_box_area:
                # save results
                tracking_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1\n" #t.score is confidence score
                )
                # Draw the bounding box with label
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                
                # Ánh xạ label từ class_id sang tên lớp
                label = tracked_objects.get(tid, -1)
                # Lưu txt
                mot_file.write(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{label},-1\n")
                # print((f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{label},-1\n"))
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, tracked_objects.get(tid, -1), type='track') # Truyền nhãn đã lưu
        
        # out_detect.write(frame_detected)
        out_tracking.write(frame_tracked)

        frame_id += 1

    cap.release()
    mot_file.close()
    # out_detect.release()
    out_tracking.release()
    print(f"Tracking results will be saved to: {mot_output_path}")
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()