from collections import defaultdict

class ObjectCounter:
    def __init__(self, line_start, line_end):
        self.counting_line = [line_start, line_end]
        self.line_y = (line_start[1] + line_end[1]) / 2
        self.object_last_position = {}  # {id: (x_center, y_center)}
        self.count_up = defaultdict(int)     # {label: count}
        self.count_down = defaultdict(int)   # {label: count}

    def update(self, object_id, bbox, label):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        prev = self.object_last_position.get(object_id, None)
        self.object_last_position[object_id] = (x_center, y_center)

        if prev is None:
            return

        prev_x, prev_y = prev

        if prev_y < self.line_y and y_center >= self.line_y:
            self.count_down[label] += 1
        elif prev_y > self.line_y and y_center <= self.line_y:
            self.count_up[label] += 1

    def get_counts(self):
        return dict(self.count_up), dict(self.count_down)
