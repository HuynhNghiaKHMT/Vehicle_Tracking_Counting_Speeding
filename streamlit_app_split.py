import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os

# Import modules from our project structure
from detector.detector import Detector
from bytetrack_tracker.byte_tracker import BYTETracker # Import TrackerArgs if defined here
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator

# Import helper functions and configurations
from util.drawing_utils import (draw_fps, draw_bbox, draw_line, draw_count, draw_zone, draw_ruler)
from util.app_state_utils import reset_session_state_and_ui
from config.constants import (
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_MIN_BOX_AREA,
    DEFAULT_CLASS_NAMES,
    DEFAULT_TRACK_THRESH, DEFAULT_TRACK_BUFFER, DEFAULT_MATCH_THRESH, DEFAULT_FUSE_SCORE,
    DEFAULT_TARGET_WIDTH_REAL, DEFAULT_TARGET_HEIGHT_REAL
)
from config.initial_values import get_default_session_state_values
from data_manager.vehicle_data_manager import VehicleDataManager

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Vivehicle")

st.title("ByteTrack Object Tracking, Counting, and Speed Estimation")
st.write("Upload a video and choose the functionalities you want to enable. Adjust line/zone points on the preview frame.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Initialize session state variables using our utility function
for key, value in get_default_session_state_values().items():
    st.session_state.setdefault(key, value)


uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# --- Video Information Placeholder ---
st.sidebar.markdown("---")
st.sidebar.subheader("Video Information")
video_info_placeholder = st.sidebar.empty() # Create a placeholder to update later
video_info_placeholder.info("Upload a video to see its dimensions.") # Initial message

# Feature Selection Checkboxes
st.sidebar.markdown("---")
st.sidebar.subheader("Select Features")
enable_counting = st.sidebar.checkbox("Object Counting", value=False)
enable_speeding = st.sidebar.checkbox("Speed Estimation", value=False)

# Detection parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")
detection_conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONF_THRESHOLD, 0.05, help="Minimum confidence score for a detection to be considered valid.")
detection_iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, DEFAULT_IOU_THRESHOLD, 0.05, help="IoU threshold for Non-Maximum Suppression (NMS) during detection.")
min_box_area = st.sidebar.slider("Minimum Box Area", 0, 200, DEFAULT_MIN_BOX_AREA, 10, help= "Minimum pixel area for a bounding box to be processed.")
class_name_input = st.sidebar.text_input("Class Names", DEFAULT_CLASS_NAMES, help="Comma-separated list of class names recognized by the model.")
class_name = [name.strip() for name in class_name_input.split(',')]

# Tracking parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Tracker Parameters (Always On)")
track_thresh = st.sidebar.slider("Track Threshold", 0.0, 1.0, DEFAULT_TRACK_THRESH, 0.05, help="Threshold for confirming a track (higher is stricter).")
track_buffer = st.sidebar.slider("Track Buffer", 1, 100, DEFAULT_TRACK_BUFFER, 1, help="Number of frames to keep a track without detection before deleting.")
match_thresh = st.sidebar.slider("Match Threshold", 0.0, 1.0, DEFAULT_MATCH_THRESH, 0.05, help="IoU threshold for matching detections to existing tracks.")
fuse_score = st.sidebar.checkbox("Fuse Score", value=DEFAULT_FUSE_SCORE, help="Whether to use fusion score for matching.")

# Placeholder for video display in main area
video_placeholder = st.empty()

# --- Preview Logic for Line/Zone Setup ---
# Check if a file is uploaded
if uploaded_file:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # If it's a new file or the temp path is invalid/missing/different
    if current_file_id != st.session_state.last_uploaded_file_id or \
       not st.session_state.temp_video_path or \
       not os.path.exists(st.session_state.temp_video_path):
        
        # Reset everything before handling the new file, then set new temp path
        reset_session_state_and_ui(video_placeholder, video_info_placeholder)
        
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        st.session_state.temp_video_path = tfile.name
        tfile.close()
        st.session_state.last_uploaded_file_id = current_file_id # Update the ID
        
        cap_preview = cv2.VideoCapture(st.session_state.temp_video_path)
        if cap_preview.isOpened():
            ret_preview, frame_preview = cap_preview.read()
            if ret_preview:
                st.session_state.preview_frame = frame_preview
                st.session_state.video_dims = (int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            else:
                st.error("Error: Could not read first frame from video. Please try another video.")
                reset_session_state_and_ui(video_placeholder, video_info_placeholder) # Reset if error
            cap_preview.release()
        else:
            st.error("Error: Could not open video file for preview. Please ensure it's a valid video format.")
            reset_session_state_and_ui(video_placeholder, video_info_placeholder) # Reset if error
    
    # Display and allow adjustment of line/zone on the preview frame IF a valid preview frame exists
    if st.session_state.preview_frame is not None and not st.session_state.run_processing:
        display_frame = st.session_state.preview_frame.copy()
        width, height = st.session_state.video_dims

        # Update video information placeholder
        video_info_placeholder.write(f"**Dimensions:** {width} x {height}")

        # Draw Ruler
        display_frame = draw_ruler(display_frame)

        # Counting Line Setup
        if enable_counting:
            st.sidebar.subheader("Counting Line Setup")
            st.sidebar.info("Adjust the line points to define the counting area.")
            
            # Initialize with default or previous values
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

        # Speed Estimation Zone Setup
        if enable_speeding:
            st.sidebar.subheader("Speed Estimation Zone Setup")
            st.sidebar.info("Adjust the four points to define the perspective transform zone.")
            st.sidebar.markdown("---")
            
            # Default points based on video dimensions (scaled examples for 1920x850, adjust as needed)
            # These values are examples and might need fine-tuning for different video resolutions/perspectives
            default_p1_x = int(width * 0.416)
            default_p1_y = int(height * 0.482)
            default_p2_x = int(width * 0.585)
            default_p2_y = int(height * 0.482)
            default_p3_x = int(width * 0.999)
            default_p3_y = int(height * 0.999)
            default_p4_x = int(width * 0.0)
            default_p4_y = int(height * 0.999)

            # Initialize with default or previous values
            if st.session_state.source_points_config is not None:
                initial_sp = st.session_state.source_points_config
                initial_p1_x, initial_p1_y = initial_sp[0]
                initial_p2_x, initial_p2_y = initial_sp[1]
                initial_p3_x, initial_p3_y = initial_sp[2]
                initial_p4_x, initial_p4_y = initial_sp[3]
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
            ])
            draw_zone(display_frame, st.session_state.source_points_config)

            st.sidebar.markdown("---")
            st.sidebar.markdown("**Actual Distance Size (for perspective transform)**")
            col_target_width, col_target_height = st.sidebar.columns(2)
            st.session_state.target_width_real_config = col_target_width.number_input("Width (m)", value=st.session_state.target_width_real_config, key="target_width_real")
            st.session_state.target_height_real_config = col_target_height.number_input("Height (m)", value=st.session_state.target_height_real_config, key="target_height_real")
        
        video_placeholder.image(display_frame, channels="BGR", use_container_width=True, caption="Video Preview (Use ruler and adjust Line/Zone in Sidebar)")
    elif st.session_state.run_processing: # If already running, don't show preview setup
        pass # Let the main processing loop handle the display
    else: # If uploaded but no valid preview frame
        st.info("Upload a video to see a preview and set up features.")
        video_info_placeholder.info("Upload a video to see its dimensions.")

# If no file is uploaded and we are not currently processing (i.e., user cleared the file or first load)
elif not uploaded_file and not st.session_state.run_processing:
    # Only reset if there was a previous temp file to clean up
    if st.session_state.temp_video_path is not None and os.path.exists(st.session_state.temp_video_path):
        reset_session_state_and_ui(video_placeholder, video_info_placeholder)
    else: # Initial load or already clean
        video_info_placeholder.info("Upload a video to see its dimensions.")


st.sidebar.markdown("---")
start_button = st.sidebar.button("Start Processing")
stop_button = st.sidebar.button("Stop Processing")

# --- Main Processing Logic (when Start button is pressed) ---
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
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Initialize Detector, Tracker, Counter, SpeedEstimator
            model = Detector("weights/best.pt") # Assuming weights are in 'weights/' folder
            args = TrackerArgs(track_thresh, track_buffer, match_thresh, fuse_score)
            tracker = BYTETracker(args)

            counter = None
            if enable_counting: # Removed redundant line_coords_config check as we handle defaults
                if st.session_state.line_coords_config:
                    counter = ObjectCounter(st.session_state.line_coords_config[0], st.session_state.line_coords_config[1])
                else:
                    # Default line if not set during preview
                    st.warning("Counting enabled but line coordinates not set during preview. Using default line for processing.")
                    counter = ObjectCounter((0, int(height * 0.75)), (width, int(height * 0.75)))

            speed_estimator = None
            if enable_speeding: # Removed redundant source_points_config check as we handle defaults
                if st.session_state.source_points_config is not None:
                    target_points = np.array([
                        [0, 0],
                        [st.session_state.target_width_real_config - 1, 0],
                        [st.session_state.target_width_real_config - 1, st.session_state.target_height_real_config - 1],
                        [0, st.session_state.target_height_real_config - 1],
                    ])
                    speed_estimator = SpeedEstimator(
                        fps=fps,
                        target_width_real=st.session_state.target_width_real_config,
                        target_height_real=st.session_state.target_height_real_config,
                        source_points=st.session_state.source_points_config,
                        target_points=target_points
                    )
                else:
                    # Default zone if not set during preview (dummy values, user warned)
                    st.warning("Speed estimation enabled but zone coordinates not set during preview. Using dummy zone for processing.")
                    speed_estimator = SpeedEstimator(fps, 1, 1, np.array([[0,0],[1,0],[1,1],[0,1]]), np.array([[0,0],[1,0],[1,1],[0,1]]))

            # --- TRY-FINALLY block for main video processing loop ---
            try:
                frame_id = 0
                tracked_objects = {} # Dictionary to store track_id -> class_id

                while st.session_state.run_processing and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.info("End of video stream.")
                        st.session_state.run_processing = False
                        break

                    frame_display = frame.copy()

                    # Draw line and zone on the processing frame
                    if enable_counting and st.session_state.line_coords_config:
                        draw_line(frame_display, st.session_state.line_coords_config)
                    if enable_speeding and st.session_state.source_points_config is not None:
                        draw_zone(frame_display, st.session_state.source_points_config)

                    # Detect
                    start_time = time.time()
                    detect_results = model.detect(frame, conf=detection_conf_threshold, iou=detection_iou_threshold)
                    detections = []
                    current_detections = [] # To store (bbox, conf, class_id) for matching

                    for result in detect_results:
                        for boxes in result.boxes:
                            label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            class_id = int(label)

                            detections.append([x1, y1, x2, y2, conf])
                            current_detections.append(((x1, y1, x2, y2), conf, class_id))

                    if detections:
                        detections = np.array(detections)
                    else:
                        detections = np.empty((0, 5))

                    # Update tracker
                    online_targets = tracker.update(detections, [height, width], [height, width])

                    end_time = time.time()
                    current_fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0

                    # Draw tracked objects
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        # t_bbox = tuple(map(int, (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])))

                        # Find the label of the tracked object based on the closest initial detection
                        matched_label = -1
                        min_distance = float('inf')
                        center_tracked_x = tlwh[0] + tlwh[2] / 2
                        center_tracked_y = tlwh[1] + tlwh[3] / 2

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
                            # If a track appears without a direct detection match on its first frame,
                            # assign a default class (e.g., 0) or handle as an unknown object.
                            # For robustness, you might want to delay drawing until a class is confirmed.
                            tracked_objects[tid] = 0 # Default to first class if no match on first frame
                        
                        label_for_draw = tracked_objects.get(tid, 0) # Get class_id for this track_id

                        if tlwh[2] * tlwh[3] > min_box_area:
                            x1, y1, w, h = map(int, tlwh)
                            x2, y2 = x1 + w, y1 + h

                            # Counting
                            if enable_counting and counter and label_for_draw != -1:
                                counter.update(tid, (x1, y1, x2, y2), label_for_draw)

                            # Speed Estimation
                            speed = None
                            if enable_speeding and speed_estimator and st.session_state.source_points_config is not None:
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Check if the object's center is inside the defined zone
                                point_in_zone = cv2.pointPolygonTest(st.session_state.source_points_config.astype(np.int32), (int(center_x), int(center_y)), False)
                                
                                if point_in_zone >= 0: # Object is inside or on the edge of the zone
                                    speed = speed_estimator.estimate_speed(tid, (center_x, center_y))
                                else:
                                    speed = None # Object is outside the zone
                            
                            draw_bbox(frame_display, tid, x1, y1, x2, y2, t.score, label_for_draw, class_name, speed, type='track')

                    # Draw FPS and Count on the frame
                    draw_fps(frame_display, current_fps)
                    
                    if enable_counting and counter:
                        draw_count(frame_display, counter, class_name)

                    # Display the frame in Streamlit
                    video_placeholder.image(frame_display, channels="BGR", use_container_width=True)

                    frame_id += 1
            
            # --- FINALLY block: Ensures cap.release() is always called and cleanup happens ---
            finally:
                if 'cap' in locals() and cap.isOpened(): # Check if cap exists and is open
                    cap.release()
                
                # Reset all relevant session state variables and UI elements via the helper
                reset_session_state_and_ui(video_placeholder, video_info_placeholder)
                st.info("Processing stopped.")


# Handle Stop button press (simply sets run_processing to False, finally block handles cleanup)
elif stop_button:
    st.session_state.run_processing = False
    # All cleanup and state resets are handled by the finally block in the main processing logic.
    # Just display a message here.
    st.info("Processing stopped by user.")


# Initial message if no video uploaded and no processing is running
if uploaded_file is None and not st.session_state.run_processing and st.session_state.preview_frame is None:
    st.info("Please upload a video file to start processing and set up features.")