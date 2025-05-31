class ObjectCounter:
    def __init__(self, line_start, line_end):
        self.counting_line = [line_start, line_end]  # [(x1, y1), (x2, y2)]
        self.line_y = (line_start[1] + line_end[1]) / 2  # Lấy y cố định để đếm
        self.object_last_position = {}  # {id: (x_center, y_center)}
        self.count_up = 0
        self.count_down = 0

    def update(self, object_id, bbox):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        prev = self.object_last_position.get(object_id, None)
        self.object_last_position[object_id] = (x_center, y_center)

        if prev is None:
            return

        prev_x, prev_y = prev

        if prev_y < self.line_y and y_center >= self.line_y:
            self.count_down += 1
        elif prev_y > self.line_y and y_center <= self.line_y:
            self.count_up += 1

    def get_counts(self):
        return self.count_up, self.count_down
