# your_project/speed_estimator/zone_speeding.py
import cv2
import numpy as np
from collections import defaultdict, deque

class ViewTransformer:
    def __init__(self, source, target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2) # Keep as float for calculations

class SpeedEstimator:
    def __init__(self, fps, target_width_real, target_height_real, source_points, target_points):
        self.fps = fps
        self.target_width_real = target_width_real
        self.target_height_real = target_height_real
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps * 2)))# Store transformed coordinates history
        self.view_transformer = ViewTransformer(source=source_points, target=target_points)
        
        # NEW: Store individual speed calculations for averaging later
        self.object_speed_history = defaultdict(list) # {id: [speed1, speed2, ...]}


    # NEW: Methods to update parameters if they change in Streamlit sidebar
    def set_fps(self, fps):
        self.fps = fps
        # Adjust deque maxlen if FPS changes significantly
        for tid in self.coordinates:
            self.coordinates[tid] = deque(self.coordinates[tid], maxlen=int(self.fps * 2))

    def set_real_world_dimensions(self, width_real, height_real):
        self.target_width_real = width_real
        self.target_height_real = height_real


    def set_zone(self, source_points, target_points=None):
        # If target_points not provided, assume a default based on real dimensions
        if target_points is None:
            target_points = np.array([
                [0, 0],
                [self.target_width_real, 0],
                [self.target_width_real, self.target_height_real],
                [0, self.target_height_real],
            ], dtype=np.float32)
        self.view_transformer = ViewTransformer(source=source_points, target=target_points)


    def transform_point(self, point_pixel):
        """Transforms a single pixel point (x,y) to real-world coordinates."""
        transformed = self.view_transformer.transform_points(np.array([point_pixel]))[0]
        return transformed
    
    def update_coordinates(self, tracker_id, transformed_point):
        self.coordinates[tracker_id].append(transformed_point)

    def estimate_speed(self, tracker_id, bbox_pixel):
        """
        Estimates speed for a given tracker_id based on its transformed coordinates history.
        Stores individual speed calculations in object_speed_history.
        """
        x1, y1, x2, y2 = bbox_pixel
        center_x_pixel = (x1 + x2) / 2
        center_y_pixel = (y1 + y2) / 2
        
        transformed_center_point = self.transform_point((center_x_pixel, center_y_pixel))
        self.update_coordinates(tracker_id, transformed_center_point)

        if len(self.coordinates[tracker_id]) >= 2:
            # Calculate speed between the last two points for instantaneous speed
            p1_real = self.coordinates[tracker_id][-1]
            p2_real = self.coordinates[tracker_id][-2]
            
            # Distance in real-world units (using Euclidean distance)
            real_distance_meters = np.linalg.norm(p1_real - p2_real)
            
            # Time elapsed for one frame
            time_elapsed_seconds = 1.0 / self.fps
            
            # DEBUG: Print real distance and time


            if time_elapsed_seconds > 0:
                speed_mps = real_distance_meters / time_elapsed_seconds
                speed_kmh = speed_mps * 3.6
               
                # NEW: Store this instantaneous speed
                self.object_speed_history[tracker_id].append(speed_kmh)
                
                return speed_kmh
        
        return None # Return None if not enough data to calculate speed

    # NEW: Method to get the full speed history for an object
    def get_speed_history(self, tracker_id):
        return self.object_speed_history.get(tracker_id, [])

    # NEW: Method to clear speed history for an object (e.g., after recording to Excel or track lost)
    def clear_object_history(self, tracker_id):
        if tracker_id in self.coordinates:

            del self.coordinates[tracker_id]
        if tracker_id in self.object_speed_history:

            del self.object_speed_history[tracker_id]