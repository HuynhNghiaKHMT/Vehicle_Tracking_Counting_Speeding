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
        return transformed_points.reshape(-1, 2).astype(int) # Chuyển sang int sau khi transform

class SpeedEstimator:
    def __init__(self, fps, target_width_real, target_height_real, source_points, target_points):
        self.fps = fps
        self.target_width_real = target_width_real
        self.target_height_real = target_height_real
        self.coordinates = defaultdict(lambda: deque(maxlen=int(fps)))# Lưu trữ tọa độ đã transform
        self.view_transformer = ViewTransformer(source=source_points, target=target_points)

    def transform_point(self, point):
            return self.view_transformer.transform_points(np.array([point]))[0]
    
    def update_coordinates(self, tracker_id, transformed_point):
        self.coordinates[tracker_id].append(transformed_point)

    def estimate_speed(self, tracker_id, current_point):
        transformed_point = self.transform_point(current_point)
        self.update_coordinates(tracker_id, transformed_point)
        if len(self.coordinates[tracker_id]) >= 2:
            point1 = self.coordinates[tracker_id][-1]
            point2 = self.coordinates[tracker_id][0]
            real_distance = abs(point1[1] - point2[1])
            time = len(self.coordinates[tracker_id]) / self.fps
            speed_mps = real_distance / time
            return speed_mps * 3.6
        
        return None