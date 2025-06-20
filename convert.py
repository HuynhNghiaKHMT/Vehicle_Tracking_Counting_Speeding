from configparser import ConfigParser
import os
# SETUP config
config = ConfigParser()
config.read('tracker.cfg')
# Đọc từ annotations ban đầu

for i in range(2,3):
    with open(f'./annotations/txt/gt_video_{i}.txt', 'r') as fin, open(config.get('annotations', f'anno_gt_{i}'), 'w') as fout:
        for line in fin:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame, tid, x, y, w, h, conf, ob_class, vis= parts[:9]
                frame = str(int(frame) - 1)  # Giảm frame ID đi 1 để match với Pred

                fout.write(f"{frame},{tid},{x},{y},{w},{h},{conf},{ob_class},{vis}\n")
