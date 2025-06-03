import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os 

from detector.detector import Detector
from bytetrack_tracker.byte_tracker import BYTETracker
from counting.line_counting_classes import ObjectCounter
from speed_estimator.zone_speeding import SpeedEstimator


# --- Helper Functions ---

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def draw_fps(frame, fps_val, width, height):
    """Draws FPS and frame dimensions on the given frame."""
    fps_text = f' FPS: {fps_val:.2f}' + f' Width: {width}' + f' Height: {height}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0) 
    background_color = (255, 255, 255) 
    text_size = cv2.getTextSize(fps_text, font, font_scale, font_thickness)[0]
    text_x = min(40, frame.shape[1] - text_size[0] - 10)
    text_y = max(50, text_size[1] + 10)
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), background_color, -1)
    cv2.putText(frame, fps_text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, label, class_names, speed=None, type='detect'):
    """Draws a bounding box and label on the frame."""
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
    text_color = (255, 255, 255)   # White text
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

def draw_line(frame, line_coords):
    """Draws a line on the frame for object counting."""
    pt1 = tuple(map(int, line_coords[0]))
    pt2 = tuple(map(int, line_coords[1]))
    cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # Green line

def draw_count(frame, counter, class_names):
    """Draws object counts on the frame."""
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

def draw_zone(frame, zone_points):
    """Draws a zone (polygon) on the frame for speed estimation."""
    pts = zone_points.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (255, 0, 0), thickness=2) # Blue color

def draw_ruler(frame):
    """Draws horizontal and vertical rulers with pixel coordinates on the frame."""
    height, width, _ = frame.shape
    ruler_color = (0, 0, 0)  # Black lines for ruler
    ruler_thickness = 2
    tick_length_major = 15
    tick_length_minor = 7
    major_interval = 100 # Every 100 pixels
    minor_interval = 50  # Every 50 pixels
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.6
    text_thickness = 2
    text_color = (0, 0, 0)      # Black text
    background_color = (255, 255, 255) # White background for text

    # Draw Horizontal Ruler (Top Edge)
    cv2.line(frame, (0, 0), (width, 0), ruler_color, ruler_thickness)
    for i in range(0, width, minor_interval):
        cv2.line(frame, (i, 0), (i, tick_length_minor), ruler_color, ruler_thickness)
        if i % major_interval == 0:
            cv2.line(frame, (i, 0), (i, tick_length_major), ruler_color, ruler_thickness)
            text = str(i)
            (text_width, text_height), baseline = cv2.getTextSize(text, text_font, text_scale, text_thickness)
            text_x_centered = i - text_width // 2 # Calculate centered X position for text

            # Draw background rectangle for text
            cv2.rectangle(frame, 
                          (text_x_centered - 5, tick_length_major + 5), 
                          (text_x_centered + text_width + 5, tick_length_major + text_height + 15), 
                          background_color, -1)
            cv2.putText(frame, text, (text_x_centered, tick_length_major + text_height + 10), text_font, text_scale, text_color, text_thickness)

    # Draw Vertical Ruler (Left Edge)
    cv2.line(frame, (0, 0), (0, height), ruler_color, ruler_thickness)
    for i in range(0, height, minor_interval):
        cv2.line(frame, (0, i), (tick_length_minor, i), ruler_color, ruler_thickness)
        if i % major_interval == 0:
            cv2.line(frame, (0, i), (tick_length_major, i), ruler_color, ruler_thickness)
            text = str(i)
            (text_width, text_height), baseline = cv2.getTextSize(text, text_font, text_scale, text_thickness)
            text_y_centered = i + text_height // 2 # Calculate centered Y position for text

            # Draw background rectangle for text
            cv2.rectangle(frame, 
                          (tick_length_major + 5, text_y_centered - text_height // 2 - 5), # Top-left corner
                          (tick_length_major + text_width + 15, text_y_centered + text_height // 2 + 5), # Bottom-right corner
                          background_color, -1)
            cv2.putText(frame, text, (tick_length_major + 10, text_y_centered + 5), text_font, text_scale, text_color, text_thickness)
    
    return frame


# --- Streamlit Application ---

st.set_page_config(layout="wide", page_title="Vivehicle")

st.title("ByteTrack Object Tracking, Counting, and Speed Estimation")
st.write("Upload a video and choose the functionalities you want to enable. Adjust line/zone points on the preview frame.")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Initialize session state variables if they don't exist
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False
if 'preview_frame' not in st.session_state:
    st.session_state.preview_frame = None
if 'video_dims' not in st.session_state:
    st.session_state.video_dims = (0, 0) # (width, height)
if 'temp_video_path' not in st.session_state:
    st.session_state.temp_video_path = None
if 'last_uploaded_file_id' not in st.session_state: # Use a unique ID for the uploaded file
    st.session_state.last_uploaded_file_id = None
if 'line_coords_config' not in st.session_state:
    st.session_state.line_coords_config = None
if 'source_points_config' not in st.session_state:
    st.session_state.source_points_config = None
if 'target_width_real_config' not in st.session_state:
    st.session_state.target_width_real_config = 32
if 'target_height_real_config' not in st.session_state:
    st.session_state.target_height_real_config = 140


uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# --- Video Information Placeholder ---
st.sidebar.markdown("---")
st.sidebar.subheader("Video Information")
# Tạo một placeholder để sau này có thể cập nhật nội dung
video_info_placeholder = st.sidebar.empty()
# Hiển thị thông báo ban đầu
video_info_placeholder.info("Upload a video to see its dimensions.")

# Feature Selection Checkboxes
st.sidebar.markdown("---")
st.sidebar.subheader("Select Features")
enable_counting = st.sidebar.checkbox("Object Counting", value=False)
enable_speeding = st.sidebar.checkbox("Speed Estimation", value=False)

# Detection parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")
detection_conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum confidence score for a detection to be considered valid.")
detection_iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.8, 0.05, help="IoU threshold for Non-Maximum Suppression (NMS) during detection.")
min_box_area = st.sidebar.slider("Minimum Box Area", 0, 200, 100, 10, help= "Minimum pixel area for a bounding box to be processed.")
class_name_input = st.sidebar.text_input("Class Names", "Motor,Car,Bus,Truck",  help="Comma-separated list of class names recognized by the model.")
class_name = [name.strip() for name in class_name_input.split(',')]

# Tracking parameters
st.sidebar.markdown("---")
st.sidebar.subheader("Tracker Parameters (Always On)")
track_thresh = st.sidebar.slider("Track Threshold", 0.0, 1.0, 0.5, 0.05, help="Threshold for confirming a track (higher is stricter).")
track_buffer = st.sidebar.slider("Track Buffer", 1, 100, 30, 1, help="Number of frames to keep a track without detection before deleting.")
match_thresh = st.sidebar.slider("Match Threshold", 0.0, 1.0, 0.8, 0.05, help="IoU threshold for matching detections to existing tracks.")
fuse_score = st.sidebar.checkbox("Fuse Score", value=True, help="Whether to use fusion score for matching.")

# Placeholder for video display in main area
video_placeholder = st.empty()


# --- Preview Logic for Line/Zone Setup ---
# Check if a file is uploaded and we are not actively processing
if uploaded_file is not None and not st.session_state.run_processing:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # If it's a new file or the temp path is invalid/missing
    if current_file_id != st.session_state.last_uploaded_file_id or \
       st.session_state.temp_video_path is None or \
       not os.path.exists(st.session_state.temp_video_path):
        
        st.session_state.last_uploaded_file_id = current_file_id # Update the ID
        
        # Clean up previous temp file if it exists
        if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
            try:
                os.remove(st.session_state.temp_video_path)
            except PermissionError:
                st.warning("Could not remove previous temp video file. It might still be in use.")
            finally:
                st.session_state.temp_video_path = None # Reset path even if removal fails
        
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        st.session_state.temp_video_path = tfile.name
        tfile.close()

        cap_preview = cv2.VideoCapture(st.session_state.temp_video_path)
        if cap_preview.isOpened():
            ret_preview, frame_preview = cap_preview.read()
            if ret_preview:
                st.session_state.preview_frame = frame_preview
                st.session_state.video_dims = (int(cap_preview.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_preview.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                # Reset config values when a new video is loaded
                st.session_state.line_coords_config = None
                st.session_state.source_points_config = None
                st.session_state.target_width_real_config = 32
                st.session_state.target_height_real_config = 140
            else:
                st.error("Error: Could not read first frame from video. Please try another video.")
                st.session_state.preview_frame = None
                st.session_state.video_dims = (0,0)
            cap_preview.release()
        else:
            st.error("Error: Could not open video file for preview. Please ensure it's a valid video format.")
            st.session_state.preview_frame = None
            st.session_state.video_dims = (0, 0)
    
    # Display and allow adjustment of line/zone on the preview frame
    if st.session_state.preview_frame is not None:
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
            st.sidebar.markdown("**actual distance size**")
            col_target_width, col_target_height = st.sidebar.columns(2)
            st.session_state.target_width_real_config = col_target_width.number_input("Width (m)", value=st.session_state.target_width_real_config, key="target_width_real")
            st.session_state.target_height_real_config = col_target_height.number_input("Height (m)", value=st.session_state.target_height_real_config, key="target_height_real")
        
        video_placeholder.image(display_frame, channels="BGR", use_container_width=True, caption="Video Preview (Use ruler and adjust Line/Zone in Sidebar)")
    else:
        # If no preview frame is available, display a message
        st.info("Upload a video to see a preview and set up features.")
        # Ensure video info placeholder is reset if no video is loaded
        video_info_placeholder.info("Upload a video to see its dimensions.")

# This block handles clearing the file uploader (e.g., user clicks 'x' on the uploaded file)
elif uploaded_file is None and st.session_state.temp_video_path is not None and not st.session_state.run_processing:
    # Ensure any active processing is stopped
    st.session_state.run_processing = False 
    
    # Try to remove the temporary file
    if os.path.exists(st.session_state.temp_video_path):
        try:
            os.remove(st.session_state.temp_video_path)
            st.session_state.temp_video_path = None
            st.session_state.last_uploaded_file_id = None # Clear the last uploaded file ID
            st.info("Previous video file removed.")
        except PermissionError:
            st.warning("Could not remove previous video file. It might still be in use. Please restart the app if issues persist.")
            st.session_state.temp_video_path = None # Reset path even if removal fails
        finally:
            # Reset all related session state variables
            st.session_state.preview_frame = None
            st.session_state.video_dims = (0, 0)
            st.session_state.line_coords_config = None
            st.session_state.source_points_config = None
            st.session_state.target_width_real_config = 32
            st.session_state.target_height_real_config = 140
            video_placeholder.empty() # Clear video display
            video_info_placeholder.info("Upload a video to see its dimensions.") # Reset placeholder
    else: # If path exists but file doesn't, just reset path
        st.session_state.temp_video_path = None
        st.session_state.last_uploaded_file_id = None
        st.session_state.preview_frame = None
        st.session_state.video_dims = (0, 0)
        st.session_state.line_coords_config = None
        st.session_state.source_points_config = None
        st.session_state.target_width_real_config = 32
        st.session_state.target_height_real_config = 140
        video_placeholder.empty()
        video_info_placeholder.info("Upload a video to see its dimensions.")

    st.info("Please upload a video file to start processing and set up features.")


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
            model = Detector("best.pt")
            args = TrackerArgs(track_thresh, track_buffer, match_thresh, fuse_score)
            tracker = BYTETracker(args)

            counter = None
            if enable_counting and st.session_state.line_coords_config:
                counter = ObjectCounter(st.session_state.line_coords_config[0], st.session_state.line_coords_config[1])
            elif enable_counting:
                st.warning("Counting enabled but line coordinates not set during preview. Using default line for processing.")
                counter = ObjectCounter((0, int(height * 0.75)), (width, int(height * 0.75)))

            speed_estimator = None
            if enable_speeding and st.session_state.source_points_config is not None:
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
            elif enable_speeding:
                st.warning("Speed estimation enabled but zone coordinates not set during preview. Using dummy zone for processing.")
                speed_estimator = SpeedEstimator(fps, 1, 1, np.array([[0,0],[1,0],[1,1],[0,1]]), np.array([[0,0],[1,0],[1,1],[0,1]]))

            # --- TRY-FINALLY block for main video processing loop ---
            try:
                frame_id = 0
                tracked_objects = {}

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
                    # Truyền các ngưỡng conf và iou từ sidebar vào phương thức detect
                    detect_results = model.detect(frame, threshold=detection_iou_threshold) 
                    detections = []
                    current_detections = []

                    for result in detect_results:
                        for boxes in result.boxes:
                            label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                            x1, y1, x2, y2 = map(int, bbox)
                            class_id = int(label)

                            if conf > detection_conf_threshold:
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
                        
                        if matched_label != -1:
                            tracked_objects[tid] = matched_label
                        elif tid not in tracked_objects:
                            tracked_objects[tid] = 0 # Default to first class if no match on first frame
                        
                        label_for_draw = tracked_objects.get(tid, 0)

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
                    draw_fps(frame_display, current_fps, width, height)
                    
                    if enable_counting and counter:
                        draw_count(frame_display, counter, class_name)

                    # Display the frame in Streamlit
                    video_placeholder.image(frame_display, channels="BGR", use_container_width=True)

                    frame_id += 1
            
            # --- FINALLY block: Ensures cap.release() is always called and cleanup happens ---
            finally:
                if 'cap' in locals() and cap.isOpened(): # Check if cap exists and is open
                    cap.release()
                st.session_state.run_processing = False # Ensure processing state is reset

                # Clean up temp file after processing is done or stopped
                if st.session_state.temp_video_path and os.path.exists(st.session_state.temp_video_path):
                    try:
                        os.remove(st.session_state.temp_video_path)
                    except PermissionError:
                        st.warning("Could not remove temp video file immediately. It might still be in use. It will be removed when Streamlit re-runs or process exits.")
                    finally: # Ensure reset path even if removal fails
                        st.session_state.temp_video_path = None
                        st.session_state.last_uploaded_file_id = None # Important: Reset uploaded file ID
                        st.session_state.preview_frame = None # Clear preview frame
                        st.session_state.video_dims = (0, 0) # Clear dimensions
                        st.session_state.line_coords_config = None # Clear configs
                        st.session_state.source_points_config = None
                        st.session_state.target_width_real_config = 32
                        st.session_state.target_height_real_config = 140
                        video_placeholder.empty() # Clear video display
                        video_info_placeholder.info("Upload a video to see its dimensions.") # Reset placeholder
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

