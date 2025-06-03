# your_project/config/constants.py

# Default detection parameters
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.8
DEFAULT_MIN_BOX_AREA = 100
DEFAULT_CLASS_NAMES = "Motor,Car,Bus,Truck" # Example class names

# Default tracker parameters
DEFAULT_TRACK_THRESH = 0.5
DEFAULT_TRACK_BUFFER = 30
DEFAULT_MATCH_THRESH = 0.8
DEFAULT_FUSE_SCORE = True

# Default speed estimation parameters
DEFAULT_TARGET_WIDTH_REAL = 32
DEFAULT_TARGET_HEIGHT_REAL = 50

# Colors for bounding boxes/labels (can be expanded)
BG_COLORS = {
    'Motor': (255, 62, 191),   # Pink
    'Car':   (41, 0, 223),     # Blue
    'Bus':   (0, 140, 255),    # Orange
    'Truck': (0, 215, 255)     # Yellow
}