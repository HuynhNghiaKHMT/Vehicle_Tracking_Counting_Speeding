# --- Imports library ---
import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os

# Import modules from our project structure
from detector.detector import Detector, Detector_RT
from bytetrack_tracker.byte_tracker import BYTETracker
from counting.line_counting_classes_xlsx import ObjectCounter
from speed_estimator.zone_speeding_xlsx import SpeedEstimator

# Import helper functions and configurations
from util.drawing_utils import (draw_fps, draw_bbox, draw_line, draw_count, draw_zone, draw_ruler)
from util.app_state_utils_xlsx import reset_session_state_and_ui
from config.constants import (
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_MIN_BOX_AREA,
    DEFAULT_CLASS_NAMES,
    DEFAULT_TRACK_THRESH, DEFAULT_TRACK_BUFFER, DEFAULT_MATCH_THRESH, DEFAULT_FUSE_SCORE,
    DEFAULT_TARGET_WIDTH_REAL, DEFAULT_TARGET_HEIGHT_REAL
)
from config.initial_values import get_default_session_state_values

# Import the VehicleDataManager
from data_manager.vehicle_data_manager import VehicleDataManager

# --- Encapsulate the configuration parameters needed for BYTETracker ---
class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

# --- Streamlit Application ---

# --- Configure Streamlit Page and Title ---
st.set_page_config(layout="wide", page_title="Vivehicle")

st.title("ByteTrack Application in Intelligent Traffic Monitoring")
st.write("Upload a video and choose the functionalities you want to enable. Adjust line/zone points on the preview frame.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# --- Initialize st.session_state and key objects ---

# Initialize session state variables using our utility function like run_processing, preview_frame, etc.
for key, value in get_default_session_state_values().items():
    st.session_state.setdefault(key, value)

# Initialize Detector, Tracker, Counter, SpeedEstimator objects once per session
# This avoids re-initialization them with every Streamlit rerun, which is inefficient.
# if 'detector_obj' not in st.session_state:
#     st.session_state.detector_obj = Detector("weights/best.pt")

st.session_state.setdefault('selected_detector_type', 'Detector')
if 'detector_obj' not in st.session_state or \
   st.session_state.get('current_detector_class_name') != st.session_state.selected_detector_type:
    
    if st.session_state.selected_detector_type == 'Detector':
        st.session_state.detector_obj = Detector("model/YOLOv8m.pt")
        st.session_state.current_detector_class_name = 'Detector'
    elif st.session_state.selected_detector_type == 'Detector_RT':
        st.session_state.detector_obj = Detector_RT("model/RT_DETR.pt")
        st.session_state.current_detector_class_name = 'Detector_RT'
        

# Initialize tracker args in session state for persistence and to pass to TrackerArgs
st.session_state.setdefault('track_thresh_config', DEFAULT_TRACK_THRESH)
st.session_state.setdefault('track_buffer_config', DEFAULT_TRACK_BUFFER)
st.session_state.setdefault('match_thresh_config', DEFAULT_MATCH_THRESH)
st.session_state.setdefault('fuse_score_config', DEFAULT_FUSE_SCORE)

if 'byte_tracker_obj' not in st.session_state:
    st.session_state.byte_tracker_obj = BYTETracker(
        TrackerArgs(st.session_state.track_thresh_config,
                    st.session_state.track_buffer_config,
                    st.session_state.match_thresh_config,
                    st.session_state.fuse_score_config)
    )

# ObjectCounter needs line_start, line_end. Update it properly when video starts with actual line_coords_config.
if 'object_counter_obj' not in st.session_state:
    st.session_state.object_counter_obj = ObjectCounter((0,0), (1,1)) 

# SpeedEstimator needs fps, real_dims, points. Update it properly when video starts with actual zone/dims/fps.
if 'speed_estimator_obj' not in st.session_state:
    st.session_state.speed_estimator_obj = SpeedEstimator(1, 1, 1, np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32), np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32))

# Initialize VehicleDataManager
if 'vehicle_data_manager' not in st.session_state:
    st.session_state.vehicle_data_manager = VehicleDataManager("vehicle_information.xlsx")

# Dictionary to store history of classes and speeds for each tracked ID
if 'vehicle_history_data' not in st.session_state:
    # Each entry: {id: {'classes': [], 'speeds': []}}
    st.session_state.vehicle_history_data = {} 

# --- Install Interface Sidebar (UI Controls) ---
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Video Information Placeholder
st.sidebar.markdown("---")
st.sidebar.subheader("Video Information")
video_info_placeholder = st.sidebar.empty() 
video_info_placeholder.info("Upload a video to see its dimensions.") 

# Detect Selection Checkboxes
st.sidebar.markdown("---")
st.sidebar.subheader("Select Detector Model")
selected_detector_option = st.sidebar.radio(
    "Choose Detection Model:",
    ('Detector (YOLOv8 Default)', 'Detector_RT (YOLOv8 Realtime)'), # Labels thân thiện hơn
    index=0, # Mặc định chọn 'Detector (YOLOv8 Default)'
    key='detector_type_radio'
)

# Map back to the actual class name string for internal use
if selected_detector_option == 'Detector (YOLOv8 Default)':
    st.session_state.selected_detector_type = 'Detector'
else: # 'Detector_RT (YOLOv8 Realtime)'
    st.session_state.selected_detector_type = 'Detector_RT'


# Feature Selection Checkboxes
st.sidebar.markdown("---")
st.sidebar.subheader("Select Features")
st.session_state.setdefault('enable_counting_config', False)
st.session_state.setdefault('enable_speeding_config', False)
enable_counting = st.sidebar.checkbox("Object Counting", value=st.session_state.enable_counting_config)
enable_speeding = st.sidebar.checkbox("Speed Estimation", value=st.session_state.enable_speeding_config)

# Detection parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")
st.session_state.setdefault('detection_conf_threshold_config', DEFAULT_CONF_THRESHOLD)
st.session_state.setdefault('detection_iou_threshold_config', DEFAULT_IOU_THRESHOLD)
st.session_state.setdefault('min_box_area_config', DEFAULT_MIN_BOX_AREA)

detection_conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, st.session_state.detection_conf_threshold_config, 0.05, help="Minimum confidence score for a detection to be considered valid.")
detection_iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, st.session_state.detection_iou_threshold_config, 0.05, help="IoU threshold for Non-Maximum Suppression (NMS) during detection.")
min_box_area = st.sidebar.slider("Minimum Box Area", 0, 2500, st.session_state.min_box_area_config, 100, help= "Minimum pixel area for a bounding box to be processed.")
# FIX: text_input expects a string, not a list.
class_name_input = st.sidebar.text_input("Class Names", "".join(DEFAULT_CLASS_NAMES), help="Comma-separated list of class names recognized by the model.")
class_name = [name.strip() for name in class_name_input.split(',')]

# Update session state for detection params
st.session_state.detection_conf_threshold_config = detection_conf_threshold
st.session_state.detection_iou_threshold_config = detection_iou_threshold
st.session_state.min_box_area_config = min_box_area


# Tracking parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Tracker Parameters (Always On)")
track_thresh = st.sidebar.slider("Track Threshold", 0.0, 1.0, st.session_state.track_thresh_config, 0.05, help="Threshold for confirming a track (higher is stricter).")
track_buffer = st.sidebar.slider("Track Buffer", 1, 100, st.session_state.track_buffer_config, 1, help="Number of frames to keep a track without detection before deleting.")
match_thresh = st.sidebar.slider("Match Threshold", 0.0, 1.0, st.session_state.match_thresh_config, 0.05, help="IoU threshold for matching detections to existing tracks.")
fuse_score = st.sidebar.checkbox("Fuse Score", value=st.session_state.fuse_score_config, help="Whether to use fusion score for matching.")

# Update session state for tracker params
st.session_state.track_thresh_config = track_thresh
st.session_state.track_buffer_config = track_buffer
st.session_state.match_thresh_config = match_thresh
st.session_state.fuse_score_config = fuse_score

# --- Logic Preview Video và Cài đặt Line/Zone ---

# Placeholder for video display in main area
video_placeholder = st.empty()

# Preview Logic for Line/Zone Setup
if uploaded_file:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if current_file_id != st.session_state.last_uploaded_file_id or \
       st.session_state.temp_video_path is None or \
       not os.path.exists(st.session_state.temp_video_path):

        reset_session_state_and_ui(video_placeholder, video_info_placeholder)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        st.session_state.temp_video_path = tfile.name
        tfile.close()
        st.session_state.last_uploaded_file_id = current_file_id

        cap_preview = cv2.VideoCapture(st.session_state.temp_video_path)
        if cap_preview.isOpened():
            ret_preview, frame_preview = cap_preview.read()
            if ret_preview:
                st.session_state.preview_frame = frame_preview
                st.session_state.video_dims = (int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            else:
                st.error("Error: Could not read first frame from video. Please try another video.")
                reset_session_state_and_ui(video_placeholder, video_info_placeholder)
            cap_preview.release()
        else:
            st.error("Error: Could not open video file for preview. Please ensure it's a valid video format.")
            reset_session_state_and_ui(video_placeholder, video_info_placeholder)

    if st.session_state.preview_frame is not None and not st.session_state.run_processing:
        display_frame = st.session_state.preview_frame.copy()
        width, height = st.session_state.video_dims

        video_info_placeholder.write(f"**Dimensions:** {width} x {height}")

        draw_ruler(display_frame)

        if enable_counting:
            st.sidebar.subheader("Counting Line Setup")
            st.sidebar.info("Adjust the line points to define the counting area.")

            default_line_y = int(height * 0.75)
            if st.session_state.line_coords_config:
                initial_lx1, initial_ly1 = st.session_state.line_coords_config[0]
                initial_lx2, initial_ly2 = st.session_state.line_coords_config[1]
            else:
                initial_lx1, initial_ly1 = 0, default_line_y
                initial_lx2, initial_ly2 = width, default_line_y

            st.sidebar.markdown("**Line Start**")
            col_start_x, col_start_y = st.sidebar.columns(2)
            line_x1 = col_start_x.number_input("X (pixel)", value=initial_lx1, min_value=0, max_value=width, key="line_x1")
            line_y1 = col_start_y.number_input("Y (pixel)", value=initial_ly1, min_value=0, max_value=height, key="line_y1")

            st.sidebar.markdown("**Line End**")
            col_end_x, col_end_y = st.sidebar.columns(2)
            line_x2 = col_end_x.number_input("X (pixel)", value=initial_lx2, min_value=0, max_value=width, key="line_x2")
            line_y2 = col_end_y.number_input("Y (pixel)", value=initial_ly2, min_value=0, max_value=height, key="line_y2")

            st.session_state.line_coords_config = [(line_x1, line_y1), (line_x2, line_y2)]
            draw_line(display_frame, st.session_state.line_coords_config)

        if enable_speeding:
            st.sidebar.subheader("Speed Estimation Zone Setup")
            st.sidebar.info("Adjust the four points to define the perspective transform zone.")
            st.sidebar.markdown("---")

            default_p1_x = int(width * 0.4)
            default_p1_y = int(height * 0.4)
            default_p2_x = int(width * 0.6)
            default_p2_y = int(height * 0.4)
            default_p3_x = int(width * 0.999)
            default_p3_y = int(height * 0.85)
            default_p4_x = int(width * 0.0)
            default_p4_y = int(height * 0.85)

            if st.session_state.source_points_config is not None and st.session_state.source_points_config.shape == (4, 2):
                initial_sp = st.session_state.source_points_config
                initial_p1_x, initial_p1_y = int(initial_sp[0][0]), int(initial_sp[0][1])
                initial_p2_x, initial_p2_y = int(initial_sp[1][0]), int(initial_sp[1][1])
                initial_p3_x, initial_p3_y = int(initial_sp[2][0]), int(initial_sp[2][1])
                initial_p4_x, initial_p4_y = int(initial_sp[3][0]), int(initial_sp[3][1])
            else:
                initial_p1_x, initial_p1_y = default_p1_x, default_p1_y
                initial_p2_x, initial_p2_y = default_p2_x, default_p2_y
                initial_p3_x, initial_p3_y = default_p3_x, default_p3_y
                initial_p4_x, initial_p4_y = default_p4_x, default_p4_y

            st.sidebar.markdown("**Point 1 (Top-Left)**")
            col_p1_x, col_p1_y = st.sidebar.columns(2)
            p1_x = col_p1_x.number_input("X (pixel)", value=initial_p1_x, min_value=0, max_value=width, key="p1_x")
            p1_y = col_p1_y.number_input("Y (pixel)", value=initial_p1_y, min_value=0, max_value=height, key="p1_y")

            st.sidebar.markdown("**Point 2 (Top-Right)**")
            col_p2_x, col_p2_y = st.sidebar.columns(2)
            p2_x = col_p2_x.number_input("X (pixel)", value=initial_p2_x, min_value=0, max_value=width, key="p2_x")
            p2_y = col_p2_y.number_input("Y (pixel)", value=initial_p2_y, min_value=0, max_value=height, key="p2_y")

            st.sidebar.markdown("**Point 3 (Bottom-Right)**")
            col_p3_x, col_p3_y = st.sidebar.columns(2)
            p3_x = col_p3_x.number_input("X (pixel)", value=initial_p3_x, min_value=0, max_value=width, key="p3_x")
            p3_y = col_p3_y.number_input("Y (pixel)", value=initial_p3_y, min_value=0, max_value=height, key="p3_y")

            st.sidebar.markdown("**Point 4 (Bottom-Left)**")
            col_p4_x, col_p4_y = st.sidebar.columns(2)
            p4_x = col_p4_x.number_input("X (pixel)", value=initial_p4_x, min_value=0, max_value=width, key="p4_x")
            p4_y = col_p4_y.number_input("Y (pixel)", value=initial_p4_y, min_value=0, max_value=height, key="p4_y")

            st.session_state.source_points_config = np.array([
                [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]
            ], dtype=np.float32)
            draw_zone(display_frame, st.session_state.source_points_config)

            st.sidebar.markdown("---")
            st.sidebar.markdown("**Actual Distance Size (for perspective transform)**")
            col_target_width, col_target_height = st.sidebar.columns(2)
            st.session_state.target_width_real_config = col_target_width.number_input("Width (m)", value=st.session_state.target_width_real_config, key="target_width_real")
            st.session_state.target_height_real_config = col_target_height.number_input("Height (m)", value=st.session_state.target_height_real_config, key="target_height_real")

        video_placeholder.image(display_frame, channels="BGR", use_container_width=True, caption="Video Preview (Use ruler and adjust Line/Zone in Sidebar)")
    elif st.session_state.run_processing:
        pass
    else:
        st.info("Upload a video to see a preview and set up features.")
        video_info_placeholder.info("Upload a video to see its dimensions.")

elif not uploaded_file and not st.session_state.run_processing and st.session_state.preview_frame is None:
    st.info("Please upload a video file to start processing and set up features.")
    video_info_placeholder.info("Upload a video to see its dimensions.")

# --- Main Processing Logic ---
st.sidebar.markdown("---")
start_button = st.sidebar.button("Start Processing")
stop_button = st.sidebar.button("Stop Processing")
manual_save_excel_button = st.sidebar.button("Save Records to Excel Now")

# --- When Start button is pressed ---
if start_button and st.session_state.temp_video_path is not None:
    st.session_state.run_processing = True
    st.write("Starting video processing...")

    if not os.path.exists(st.session_state.temp_video_path):
        st.error("Video file not found. Please re-upload the video.")
        st.session_state.run_processing = False
    else:
        cap = cv2.VideoCapture(st.session_state.temp_video_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            st.session_state.run_processing = False
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30

            st.session_state.byte_tracker_obj = BYTETracker(
                TrackerArgs(track_thresh, track_buffer, match_thresh, fuse_score)
            )

            if enable_counting:
                if st.session_state.line_coords_config:
                    st.session_state.object_counter_obj = ObjectCounter(st.session_state.line_coords_config[0], st.session_state.line_coords_config[1])
                else:
                    st.warning("Counting enabled but line coordinates not set during preview. Using default line for processing.")
                    st.session_state.object_counter_obj = ObjectCounter((0, int(height * 0.75)), (width, int(height * 0.75)))
            
            if enable_speeding:
                if st.session_state.source_points_config is not None and st.session_state.source_points_config.shape == (4, 2):
                    target_points_for_estimator = np.array([
                        [0, 0],
                        [st.session_state.target_width_real_config, 0],
                        [st.session_state.target_width_real_config, st.session_state.target_height_real_config],
                        [0, st.session_state.target_height_real_config],
                    ], dtype=np.float32)

                    st.session_state.speed_estimator_obj.set_fps(fps)
                    st.session_state.speed_estimator_obj.set_real_world_dimensions(st.session_state.target_width_real_config, st.session_state.target_height_real_config)
                    st.session_state.speed_estimator_obj.set_zone(st.session_state.source_points_config, target_points_for_estimator)
                else:
                    st.warning("Speed estimation enabled but zone coordinates not set during preview. Using dummy zone for processing.")
                    st.session_state.speed_estimator_obj = SpeedEstimator(fps, 1, 1, np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32), np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32))
            
            try:
                frame_id = 0
                prev_frame_time = time.time()

                while st.session_state.run_processing and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.info("End of video stream.")
                        st.session_state.run_processing = False
                        break
                    

                    frame_display = frame.copy()

                    if enable_counting and st.session_state.line_coords_config:
                        draw_line(frame_display, st.session_state.line_coords_config)
                    if enable_speeding and st.session_state.source_points_config is not None:
                        draw_zone(frame_display, st.session_state.source_points_config)

                    # Detect 
                    detect_results = st.session_state.detector_obj.detect(frame, conf=detection_conf_threshold, iou=detection_iou_threshold)
                    detections_for_tracker = []
                    current_detections_with_class = []

                    for result in detect_results:
                        for boxes in result.boxes:
                            label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            

                            if (x2 - x1) * (y2 - y1) >= min_box_area:
                                detections_for_tracker.append(np.array([x1, y1, x2, y2, conf]))
                                current_detections_with_class.append(((x1, y1, x2, y2), conf, int(label)))


                    if detections_for_tracker:
                        detections_for_tracker = np.array(detections_for_tracker)
                    else:
                        detections_for_tracker = np.empty((0, 5))
                    
                    # Update tracker with new detections
                    online_targets = st.session_state.byte_tracker_obj.update(detections_for_tracker, frame.shape[:2], (height, width))

                    new_frame_time = time.time()
                    current_fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
                    prev_frame_time = new_frame_time
                    draw_fps(frame_display, current_fps)

                    active_track_ids = {t.track_id for t in online_targets}

                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        

                        x1, y1, w, h = map(int, tlwh)
                        x2, y2 = x1 + w, y1 + h

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width - 1, x2)
                        y2 = min(height - 1, y2)
                        

                        matched_class_id = -1
                        min_center_distance = float('inf')
                        
                        current_bbox_center_x = x1 + w / 2
                        current_bbox_center_y = y1 + h / 2

                        for det_bbox, det_conf, det_cls_id in current_detections_with_class:
                            det_center_x = (det_bbox[0] + det_bbox[2]) / 2
                            det_center_y = (det_bbox[1] + det_bbox[3]) / 2
                            distance = (current_bbox_center_x - det_center_x)**2 + (current_bbox_center_y - det_center_y)**2
                            if distance < min_center_distance:
                                min_center_distance = distance
                                matched_class_id = det_cls_id
                        
                        if matched_class_id == -1 and tid in st.session_state.vehicle_history_data:
                            if st.session_state.vehicle_history_data[tid]['classes']:
                                last_class_name = st.session_state.vehicle_history_data[tid]['classes'][-1]
                                try:
                                    matched_class_id = class_name.index(last_class_name)
                                except ValueError:
                                    matched_class_id = 0
                            else:
                                matched_class_id = 0
                        elif matched_class_id == -1:
                            matched_class_id = 0

                        object_class_name = class_name[matched_class_id]


                        if tid not in st.session_state.vehicle_history_data:
                            st.session_state.vehicle_history_data[tid] = {
                                'classes': [],
                                'speeds': []
                            }
                        
                        st.session_state.vehicle_history_data[tid]['classes'].append(object_class_name)
                        

                        # Estimated speed
                        speed_kmh = None
                        if enable_speeding and st.session_state.speed_estimator_obj:
                            current_bbox_center_x_int = int(current_bbox_center_x)
                            current_bbox_center_y_int = int(current_bbox_center_y)

                            if st.session_state.source_points_config is not None and st.session_state.source_points_config.shape == (4, 2):
                                point_in_zone = cv2.pointPolygonTest(
                                    st.session_state.source_points_config.astype(np.int32),
                                    (current_bbox_center_x_int, current_bbox_center_y_int),
                                    False
                                )
                                
                                if point_in_zone >= 0: # Object is inside or on the edge of the zone
                                    speed_kmh = st.session_state.speed_estimator_obj.estimate_speed(tid, (x1, y1, x2, y2))
                                    if speed_kmh is not None and speed_kmh > 0:
                                        st.session_state.vehicle_history_data[tid]['speeds'].append(speed_kmh)
                                else:
                                    # Object is outside the zone, clear its speed history to prevent old speeds from affecting average
                                    # This ensures speed calculation restarts if it re-enters the zone.
                                    st.session_state.speed_estimator_obj.clear_object_history(tid)
                                    if tid in st.session_state.vehicle_history_data:
                                        st.session_state.vehicle_history_data[tid]['speeds'].clear()
                            # else: No speed estimation if zone is not configured correctly.

                        # Count objects and export to Excel
                        if enable_counting and st.session_state.object_counter_obj:
                            st.session_state.object_counter_obj.update(tid, (x1, y1, x2, y2), matched_class_id)

                            if st.session_state.object_counter_obj.has_just_crossed(tid):
                                
                                success = st.session_state.vehicle_data_manager.add_vehicle_record(
                                    vehicle_id=tid,
                                    vehicle_class_history=st.session_state.vehicle_history_data[tid]['classes'],
                                    vehicle_speed_history=st.session_state.vehicle_history_data[tid]['speeds']
                                )
                                if success:
                                    st.session_state.object_counter_obj.mark_as_recorded(tid)
                                    
                                    if tid in st.session_state.vehicle_history_data:
                                        del st.session_state.vehicle_history_data[tid]
                                    st.session_state.speed_estimator_obj.clear_object_history(tid)


                        draw_bbox(frame_display, tid, x1, y1, x2, y2, t.score, matched_class_id, class_name, speed_kmh, type='track')

                    ids_to_remove = [
                        tid for tid in st.session_state.vehicle_history_data
                        if tid not in active_track_ids
                    ]
                    for tid in ids_to_remove:
                        # Check if it crossed but was lost before being recorded, record it now
                        if enable_counting and st.session_state.object_counter_obj.object_excel_status.get(tid, {}).get('crossed_flag', False) and \
                           not st.session_state.object_counter_obj.object_excel_status.get(tid, {}).get('recorded_for_excel', False):

                            st.session_state.vehicle_data_manager.add_vehicle_record(
                                vehicle_id=tid,
                                vehicle_class_history=st.session_state.vehicle_history_data[tid]['classes'],
                                vehicle_speed_history=st.session_state.vehicle_history_data[tid]['speeds']
                            )
                            st.session_state.object_counter_obj.mark_as_recorded(tid)
                        
                        del st.session_state.vehicle_history_data[tid]
                        st.session_state.speed_estimator_obj.clear_object_history(tid)
                        st.session_state.object_counter_obj.remove_object_status(tid)


                    if enable_counting and st.session_state.object_counter_obj:
                        draw_count(frame_display, st.session_state.object_counter_obj, class_name)

                    video_placeholder.image(frame_display, channels="BGR", use_container_width=True)

                    frame_id += 1

            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()

                st.session_state.vehicle_data_manager.save_to_excel()

                reset_session_state_and_ui(video_placeholder, video_info_placeholder)
                st.info("Processing stopped and records saved (if any).")

# When Stop button is pressed
elif stop_button:
    st.session_state.run_processing = False
    st.info("Processing stopped by user. Records will be saved.")

# When Export button is pressed
if manual_save_excel_button:
    st.session_state.vehicle_data_manager.save_to_excel()
    st.success("Records saved to Excel!")

