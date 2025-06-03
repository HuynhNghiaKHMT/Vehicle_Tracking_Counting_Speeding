import motmetrics as mm
from configparser import ConfigParser

# SETUP config
config = ConfigParser()
config.read('tracker.cfg')

def evaluate_mot(gt_path, pred_path):
    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=1)
    pred = mm.io.loadtxt(pred_path, fmt='mot15-2D', min_confidence=0)

        # ✂️ CẮT pred về cùng frame range với gt
    max_gt_frame = gt.index.get_level_values(0).max()
    pred = pred[pred.index.get_level_values(0) <= max_gt_frame]

    # print("Frame cuối cùng GT:", gt.index.get_level_values(1).max())
    # print("Frame cuối cùng Pred:", pred.index.get_level_values(1).max())

    # print("GT columns:", gt.columns)
    # print("GT sample:\n", gt.head())

    # print("Pred columns:", pred.columns)
    # print("Pred sample:\n", pred.head())
    acc = mm.MOTAccumulator(auto_id=True)

     # Lấy index FrameId và Id
    gt_frames = gt.groupby(level='FrameId')
    pred_frames = pred.groupby(level='FrameId')
    all_frames = sorted(set(gt.index.get_level_values('FrameId')).union(pred.index.get_level_values('FrameId')))

    for frame in all_frames:
        gt_frame = gt_frames.get_group(frame) if frame in gt_frames.groups else None
        pred_frame = pred_frames.get_group(frame) if frame in pred_frames.groups else None
        

        n_gt = len(gt_frame) if gt_frame is not None else 0
        n_pred = len(pred_frame) if pred_frame is not None else 0
        # print(f"Frame {frame}: GT objects = {n_gt}, Pred objects = {n_pred}")

        gt_ids = gt_frame.index.get_level_values('Id').tolist() if gt_frame is not None else []
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values if gt_frame is not None else []

        pred_ids = pred_frame.index.get_level_values('Id').tolist() if pred_frame is not None else []
        pred_boxes = pred_frame[['X', 'Y', 'Width', 'Height']].values if pred_frame is not None else []


        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)
        # print(f"[Frame {frame}] GT_IDs: {gt_ids} - Pred_IDs: {pred_ids}")
        # print(f"IOU Matrix:\n{distances}")  


    mh = mm.metrics.create()
    summary = mh.compute(acc,
                         metrics=['mota', 'motp', 'idf1', 'idtp', 'idfn', 'idfp', 'num_switches'],
                         name='TrackingEval')

    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

# Gọi hàm với file
for i in range(2, 6):
    print(f"Đang đánh giá video {i}...")
    pred_path =  config.get('annotations', f'anno_pred_{i}_RT_BT')  # Đường dẫn đến file dự đoán
    gt_path =  config.get('annotations', f'anno_gt_{i}')  # Đường dẫn đến file ground truth

    evaluate_mot(gt_path, pred_path)
