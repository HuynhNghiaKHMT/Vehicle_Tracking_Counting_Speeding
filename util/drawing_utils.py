# your_project/utils/drawing_utils.py
import cv2
import numpy as np
import datetime
from config.constants import BG_COLORS # Import colors from config

def draw_fps(frame, fps_val):
    """Draws FPS and frame dimensions on the given frame."""
    current_time = datetime.datetime.now().strftime("%H:%M:%S") # Get current time
    fps_text = f' FPS: {fps_val:.2f} | Time: {current_time}' # Combine FPS and time
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
    
    background_color = BG_COLORS.get(class_names[label], (0, 0, 0)) # Default to black

    cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 255, 255) # White text
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