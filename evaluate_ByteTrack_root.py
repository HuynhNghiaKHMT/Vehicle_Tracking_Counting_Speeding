
import cv2
import numpy as np
import time
from configparser import ConfigParser
from detector.detector import Detector
from tracker.byte_tracker import BYTETracker
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator

# SETUP config
config = ConfigParser()
config.read('tracker.cfg')

# SETUP video
cap = cv2.VideoCapture(config.get('video', 'video_path5'))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out_tracking = cv2.VideoWriter(config.get('video', 'video_out_tracking1'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(config.get('video', 'video_out_detect1'), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

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

def draw_line(frame, counter):
    # print("counter.line_y:", counter.line_y)
    pt1 = tuple(map(int, counter.counting_line[0]))
    pt2 = tuple(map(int, counter.counting_line[1]))
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

def draw_count(frame, counter):
    # up, down = counter.get_counts()
    # cv2.putText(frame_tracked, f'Up: {up}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame_tracked, f'Down: {down}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    count_up, count_down = counter.get_counts()
    start_y = 50
    margin = 30
    frame_width = frame.shape[1]

    for label in sorted(set(count_up.keys()) | set(count_down.keys())):
        label_name = class_name[label]
        up = count_up.get(label, 0)
        down = count_down.get(label, 0)

        text = f"{label_name}: IN {down} OUT {up}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 0) 
        background_color = (255, 255, 255)
        text_x, text_y = width-400, start_y
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2) # Tính kích thước text
        x_pos = frame_width - text_width - margin # Căn phải

        cv2.rectangle(frame, (x_pos - 5, text_y - text_size[1] - 5), (x_pos + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x_pos, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        start_y += 50

def draw_zone(frame, zone):
    pts = zone.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))# Thay đổi hình dạng của mảng để phù hợp với hàm cv2.polylines
    cv2.polylines(frame, [pts], True, (255, 0, 0), thickness=2) # Màu xanh lá cây, độ dày đường là 2
 
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
    tracker = BYTETracker(args)
    frame_id = 0
    tracking_results = [] # store tracking results for eval, debug,...
    tracked_objects = {} # Dictionary to store tracked objects with their labels

    line_start=(0, height * 0.75)
    line_end=(width, height * 0.75)
    counter = ObjectCounter(line_start, line_end)

    target_width = 32
    target_height = 140
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # source = np.array([[1252, 787],[2298, 803],[5039, 2159],[-550, 2159]]) # video_04
    source = np.array(((800, 410), (1125, 410), (1920, 850), (0, 850))) # video_05
    target = np.array([[0, 0],[target_width - 1, 0],[target_width - 1, target_height - 1],[0, target_height - 1],])
    speed_estimator = SpeedEstimator(fps=fps, target_width_real=target_width, target_height_real=target_height, source_points=source, target_points=target)


    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        # Draw line
        draw_line(frame, counter)

        # Draw zone
        draw_zone(frame, source)
        
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
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n" #t.score is confidence score
                )
                # Draw the bounding box with label
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h

                # Counting using the ObjectCounter classs
                # counter.update(tid, (x1, y1, x2, y2)) # total count
                label = tracked_objects.get(tid, -1)
                if label != -1:
                    counter.update(tid, (x1, y1, x2, y2), label) #  count per class
                
                # Estimate speed using the SpeedEstimator classs
                speed = speed_estimator.estimate_speed(tid, (center_tracked_x, center_tracked_y))


                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, tracked_objects.get(tid, -1), speed, type='track') # Truyền nhãn đã lưu
        
        # Write and display the frame
        draw_fps(frame_detected, fps)
        draw_fps(frame_tracked, fps)
        draw_count(frame_tracked, counter)
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