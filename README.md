# Yolo + ByteTracker + Counting + Speeding Application (CS338)
Using raw code from ByteTrack [repo](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker) for educational purpose

# Đánh giá
- convert.py: chuyển định dạng file annotations thu từ annotation tool về giống định dạng thu được từ file model --> Thu được file anno_gt_videox (với x là stt của video)
- run_ByteTrack.py: dùng model để tracking --> Thu được file video_x_predictions (với x là stt của video)
- evaluate_ByteTrack.py: tiến hành đánh giá trên các độ đo: MOTA, MOTP, IDF1.


