import cv2
import pandas as pd
from configparser import ConfigParser

# Hàm load MOT file thành DataFrame
def load_mot_file(path):
    cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    df = pd.read_csv(path, header=None, names=cols)
    return df

# Đọc config
config = ConfigParser()
config.read('tracker.cfg')

# Đường dẫn
pred_path = "annotations/predicted/RT_DETR_S/video_3_predictions.txt"  # file predict
gt_path = config.get('annotations', 'anno_gt_3')      # file gt
video_path = config.get('video', 'video_path3')       # video gốc

# Load dữ liệu annotation
pred_df = load_mot_file(pred_path)
gt_df = load_mot_file(gt_path)

# Mở video gốc
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {width}x{height}")
fps = int(cap.get(cv2.CAP_PROP_FPS))

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lọc bbox GT và predict frame hiện tại
    gt_boxes = gt_df[gt_df['frame'] == frame_id]
    pred_boxes = pred_df[pred_df['frame'] == frame_id]

    # Vẽ bbox GT (màu xanh lá)
    for _, row in gt_boxes.iterrows():
        x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
        obj_id = int(row['id'])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'GT {obj_id}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Vẽ bbox predict (màu đỏ)
    for _, row in pred_boxes.iterrows():
        x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
        obj_id = int(row['id'])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'PR {obj_id}', (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Hiển thị frame
    cv2.imshow("Tracking Viewer", frame)
    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == 27:  # ESC để thoát
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()
