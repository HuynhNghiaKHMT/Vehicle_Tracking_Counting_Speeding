import streamlit as st
import cv2
import numpy as np
import time
import tempfile

# Import your actual classes (ensure these files are correctly placed)
from detector.detector import Detector
from tracker.byte_tracker import BYTETracker
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator


# --- Helper Functions (from your original code) ---

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def draw_fps(frame, fps_val, width, height):
    fps_text = f' FPS: {fps_val:.2f}' + f' Width: {width}' + f' Height: {height}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0) # Black
    background_color = (255, 255, 255) # White
    text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]
    text_x, text_y = 40, 50
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), background_color, -1)
    cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, label, class_names, speed=None, type='detect'):
    bg_colors = {
        'Motor': (255, 62, 191),   # Pink
        'Car':   (41, 0, 223),     # Red
        'Bus':   (0, 140, 255),    # Orange
        'Truck': (0, 215, 255)     # Yellow
    }

    background_color = bg_colors.get(class_names[label], (0, 0, 0)) # Default to black

    cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15

    if type == "detect":
        text = f"{class_names[label]}:{conf:.2f}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)
    elif type == "track":
        text = f'#{id}-{class_names[label]}'
        if speed is not None:
            text += f':{int(speed)}km/h'

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        cv2.rectangle(frame, (x1 - 5, text_y - text_size[1] - 5), (x1 + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x1, text_y), font, font_scale, text_color, font_thickness)

def draw_line(frame, counter):
    pt1 = tuple(map(int, counter.counting_line[0]))
    pt2 = tuple(map(int, counter.counting_line[1]))
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

def draw_count(frame, counter, class_names):
    count_up, count_down = counter.get_counts()
    start_y = 50
    margin = 30
    frame_width = frame.shape[1]

    all_labels = sorted(set(count_up.keys()) | set(count_down.keys()))

    for label in all_labels:
        label_name = class_names[label]
        up = count_up.get(label, 0)
        down = count_down.get(label, 0)

        text = f"{label_name}: IN {down} OUT {up}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 0)
        background_color = (255, 255, 255)

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        x_pos = frame_width - text_width - margin
        text_y = start_y

        cv2.rectangle(frame, (x_pos - 5, text_y - text_height - 5), (x_pos + text_width + 5, text_y + 5), background_color, -1)
        cv2.putText(frame, text, (x_pos, text_y), font, font_scale, text_color, font_thickness)
        start_y += 50

def draw_zone(frame, zone):
    pts = zone.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (255, 0, 0), thickness=2) # Blue color

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="ByteTrack Object Tracking & Counting")

st.title("ByteTrack Object Tracking, Counting, and Speed Estimation")
st.write("Upload a video and choose the functionalities you want to enable.")

# Sidebar for configuration
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# --- Feature Selection Checkboxes ---
st.sidebar.subheader("Select Features")
enable_tracking = st.sidebar.checkbox("Enable Object Tracking", value=True)
enable_counting = st.sidebar.checkbox("Enable Object Counting", value=False)
enable_speeding = st.sidebar.checkbox("Enable Speed Estimation", value=False)

# Conditional display of parameters based on feature selection
if enable_tracking:
    st.sidebar.subheader("Tracker Parameters")
    track_thresh = st.sidebar.slider("Track Threshold", 0.0, 1.0, 0.5, 0.05)
    track_buffer = st.sidebar.slider("Track Buffer", 1, 100, 30, 1)
    match_thresh = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.8, 0.05)
    fuse_score = st.sidebar.checkbox("Fuse Score", value=True)
else:
    # Set default values if tracking is disabled, though they won't be used
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = True

st.sidebar.subheader("Detection Parameters")
min_box_area = st.sidebar.slider("Minimum Box Area", 0, 500, 100, 10)
class_name_input = st.sidebar.text_input("Class Names (comma-separated)", "Motor,Car,Bus,Truck")
class_name = [name.strip() for name in class_name_input.split(',')]


if enable_speeding:
    st.sidebar.subheader("Speed Estimation Parameters")
    target_width_real = st.sidebar.number_input("Target Width Real (units)", value=32)
    target_height_real = st.sidebar.number_input("Target Height Real (units)", value=140)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Perspective Transform Zone Points (x,y)**")
    st.sidebar.info("Adjust these based on your video for accurate speed estimation.")

    p1_x = st.sidebar.number_input("Zone Point 1 X", value=800)
    p1_y = st.sidebar.number_input("Zone Point 1 Y", value=410)
    p2_x = st.sidebar.number_input("Zone Point 2 X", value=1125)
    p2_y = st.sidebar.number_input("Zone Point 2 Y", value=410)
    p3_x = st.sidebar.number_input("Zone Point 3 X", value=1920)
    p3_y = st.sidebar.number_input("Zone Point 3 Y", value=850)
    p4_x = st.sidebar.number_input("Zone Point 4 X", value=0)
    p4_y = st.sidebar.number_input("Zone Point 4 Y", value=850)

    source_points = np.array([
        [p1_x, p1_y],
        [p2_x, p2_y],
        [p3_x, p3_y],
        [p4_x, p4_y]
    ])

    target_points = np.array([
        [0, 0],
        [target_width_real - 1, 0],
        [target_width_real - 1, target_height_real - 1],
        [0, target_height_real - 1],
    ])
else:
    # Set default/dummy values if speeding is disabled
    source_points = np.array([[0,0],[1,0],[1,1],[0,1]]) # Dummy points
    target_points = np.array([[0,0],[1,0],[1,1],[0,1]]) # Dummy points
    target_width_real = 1
    target_height_real = 1


st.sidebar.markdown("---")
start_button = st.sidebar.button("Start Processing")
stop_button = st.sidebar.button("Stop Processing")

# Placeholder for video display
video_placeholder = st.empty()
count_placeholder = st.empty()


if start_button and uploaded_file is not None:
    st.session_state.run_processing = True
    st.write("Processing video...")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        st.session_state.run_processing = False
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        model = Detector("best.pt") # Detector is always initialized for basic detection
        
        # Initialize tracker, counter, speed_estimator only if enabled
        tracker = None
        if enable_tracking:
            args = TrackerArgs(track_thresh, track_buffer, match_thresh, fuse_score)
            tracker = BYTETracker(args)

        counter = None
        if enable_counting:
            line_start = (0, int(height * 0.75))
            line_end = (width, int(height * 0.75))
            counter = ObjectCounter(line_start, line_end)

        speed_estimator = None
        if enable_speeding:
            speed_estimator = SpeedEstimator(
                fps=fps,
                target_width_real=target_width_real,
                target_height_real=target_height_real,
                source_points=source_points,
                target_points=target_points
            )

        frame_id = 0
        tracked_objects = {} # Dictionary to store tracked objects with their labels

        while st.session_state.run_processing:
            ret, frame = cap.read()
            if not ret:
                st.info("End of video stream.")
                break

            frame_display = frame.copy()

            # Draw line and zone only if enabled
            if enable_counting and counter:
                draw_line(frame_display, counter)
            if enable_speeding: # Zone is for speed estimation
                draw_zone(frame_display, source_points)

            # Detect (always runs to get initial detections)
            start_time = time.time()
            detect_results = model.detect(frame)
            detections = []
            current_detections = []

            for result in detect_results:
                # Ensure result.boxes is iterable. If Detector was replaced with actual, this loop might need adjustment.
                # Assuming original Detector.detect returns an iterable of boxes.
                for boxes in result.boxes:
                    label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]

                    x1, y1, x2, y2 = map(int, bbox)
                    class_id = int(label)

                    if conf > 0.5:
                        detections.append([x1, y1, x2, y2, conf])
                        current_detections.append(((x1, y1, x2, y2), conf, class_id))

                        # Optional: Draw raw detections if tracking is OFF or if you want to see both
                        # if not enable_tracking:
                        #     draw_bbox(frame_display, None, x1, y1, x2, y2, conf, class_id, class_name, None, type='detect')

            if detections:
                detections = np.array(detections)
            else:
                detections = np.empty((0, 5))

            online_targets = []
            if enable_tracking and tracker:
                online_targets = tracker.update(detections, [height, width], [height, width])
            else:
                # If tracking is disabled, treat current detections as "tracked" for drawing purposes
                # Assign dummy IDs for display if no tracker is active
                for i, ((x1,y1,x2,y2), conf, label) in enumerate(current_detections):
                    class DummyTrackedObject:
                        def __init__(self, tlwh, track_id, score):
                            self.tlwh = tlwh
                            self.track_id = track_id
                            self.score = score
                    # Create a dummy unique ID for each detection to draw it
                    dummy_tid = f"det_{frame_id}_{i}"
                    dummy_tlwh = [x1, y1, x2-x1, y2-y1]
                    online_targets.append(DummyTrackedObject(dummy_tlwh, dummy_tid, conf))
                    # Store label for these dummy tracked objects
                    tracked_objects[dummy_tid] = label


            end_time = time.time()
            current_fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

            # Draw tracked objects or detections
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                t_bbox = tuple(map(int, (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])))

                # If tracking is enabled, try to match detection labels
                if enable_tracking:
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
                    
                    if matched_label != -1:
                        tracked_objects[tid] = matched_label
                    elif tid not in tracked_objects:
                        tracked_objects[tid] = 0 # Default to first class if no match on first frame
                
                # Use stored label
                label_for_draw = tracked_objects.get(tid, 0) # Fallback to 0 if not found (shouldn't happen with logic above)

                if tlwh[2] * tlwh[3] > min_box_area:
                    x1, y1, w, h = map(int, tlwh)
                    x2, y2 = x1 + w, y1 + h

                    # Counting
                    if enable_counting and counter and label_for_draw != -1:
                        counter.update(tid, (x1, y1, x2, y2), label_for_draw)

                    # Speed Estimation
                    speed = None
                    if enable_speeding and speed_estimator:
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        speed = speed_estimator.estimate_speed(tid, (center_x, center_y))
                    
                    # Draw bbox for tracked/detected objects
                    draw_bbox(frame_display, tid, x1, y1, x2, y2, t.score, label_for_draw, class_name, speed, type='track')

            # Draw FPS and Count on the frame
            draw_fps(frame_display, current_fps, width, height)
            
            if enable_counting and counter:
                draw_count(frame_display, counter, class_name)

            # Display the frame in Streamlit
            video_placeholder.image(frame_display, channels="BGR", use_container_width=True) # Updated to use_container_width

            frame_id += 1

        cap.release()
        tfile.close() # Close and delete the temporary file

elif stop_button:
    st.session_state.run_processing = False
    st.info("Processing stopped.")
    video_placeholder.empty()
    count_placeholder.empty()

# Initialize session state for running status
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False

if uploaded_file is None and not st.session_state.run_processing:
    st.info("Please upload a video file to start processing and select features.")

