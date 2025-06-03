# Yolo + ByteTracker + Counting + Speeding Application (CS338)

- This repo implement a simple MOT on a selt-collected vehicle dataset

- Using raw code from ByteTrack [repo](https://github.com/ifzhang/ByteTrack/tree/main/yolox/tracker) for educational purpose

# Demo

```
pip install -r requirements.txt
python demo_ByteTrack.py
```

# Demo with streamlit application

```
pip install -r requirements_streamlit.txt
python streamlit_app.py
```

# Evaluate

- Convert.py: convert the annotations from the annotation tool to the same format obtained from the file model --> Obtain the anno_gt_videox file (where x is the video status)
- run_ByteTrack.py: use the model to track --> Obtain the anno_pred_videox file (where x is the video status)
- evaluate_ByteTrack.py: conduct evaluation according to the metrics: MOTA, MOTP, IDF1.
