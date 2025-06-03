# your_project/counting/line_counting_classes.py
from collections import defaultdict

class ObjectCounter:
    """Count the number of objects (vehicles) passing through an imaginary line."""
    def __init__(self, line_start, line_end):
        self.counting_line = [line_start, line_end]
        self.line_y = (line_start[1] + line_end[1]) / 2
        self.object_last_position = {}  # {id: (x_center, y_center)}
        self.count_up = defaultdict(int)    # {label: count}
        self.count_down = defaultdict(int)  # {label: count}

        self.line_x_min = min(line_start[0], line_end[0])
        self.line_x_max = max(line_start[0], line_end[0])
        
        # Track crossing status for Excel export
        # {id: {'crossed_flag': False, 'recorded_for_excel': False}}
        self.object_excel_status = {} 

    # Update object position and perform counting.
    def update(self, object_id, bbox, label):
        
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        prev_pos = self.object_last_position.get(object_id, None)
        self.object_last_position[object_id] = (x_center, y_center)

        # Initialize status for new objects
        if object_id not in self.object_excel_status:
            self.object_excel_status[object_id] = {'crossed_flag': False, 'recorded_for_excel': False}


        if prev_pos is None:
            return

        prev_x, prev_y = prev_pos
        

        if x_center >= self.line_x_min and x_center <= self.line_x_max:
            # Check for crossing
            # A crossing event occurs when the object's center crosses the line_y
            # Its 'crossed_flag' is not already set to True.
            if prev_y < self.line_y and y_center >= self.line_y:
                # Crossed downwards (OUT)
                self.count_down[label] += 1
                if not self.object_excel_status[object_id]['crossed_flag']: # Only set flag if it's a new crossing
                    self.object_excel_status[object_id]['crossed_flag'] = True 

            elif prev_y > self.line_y and y_center <= self.line_y:
                # Crossed upwards (IN)
                self.count_up[label] += 1
                if not self.object_excel_status[object_id]['crossed_flag']: # Only set flag if it's a new crossing
                    self.object_excel_status[object_id]['crossed_flag'] = True 
       
        # Reset crossed_flag and recorded_for_excel if object moves far away from the line
        # Allows re-recording if it crosses back later. Adjust threshold (e.g., 50 pixels) as needed.

        if self.object_excel_status[object_id]['crossed_flag'] and \
           abs(y_center - self.line_y) > 50: # If it moved 50 pixels away from the line
            if self.object_excel_status[object_id]['recorded_for_excel']: # Only reset if already recorded
                self.object_excel_status[object_id]['crossed_flag'] = False
                self.object_excel_status[object_id]['recorded_for_excel'] = False # Allow re-recording

    def get_counts(self):
        return dict(self.count_up), dict(self.count_down)

    # Check and manage crossing status for Excel export
    def has_just_crossed(self, object_id):
        """
        Returns True if the object has crossed the line and has not yet been recorded for Excel.
        This flag is set by the update method when a crossing is detected.
        """
        if object_id in self.object_excel_status:
            result = self.object_excel_status[object_id]['crossed_flag'] and \
                     not self.object_excel_status[object_id]['recorded_for_excel']
            
            return result
        return False

    def mark_as_recorded(self, object_id):
        """
        Marks an object as recorded for Excel export.
        """
        if object_id in self.object_excel_status:
            self.object_excel_status[object_id]['recorded_for_excel'] = True

    def remove_object_status(self, object_id):
        """
        Removes an object's status when it's no longer tracked (e.g., leaves frame).
        """
        if object_id in self.object_excel_status:
            del self.object_excel_status[object_id]
        if object_id in self.object_last_position:
            del self.object_last_position[object_id]