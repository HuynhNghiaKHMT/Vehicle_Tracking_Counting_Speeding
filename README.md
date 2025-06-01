# Simple Yolo + ByteTracker Application (CS431)
- This repo implement a simple MOT on a selt-collected vehicle dataset

- Using raw code for educational purpose
+ Sort [repo](https://github.com/abewley/sort)
+ DeepSort [repo](https://github.com/nwojke/deep_sort)
+ Sort [repo](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/sort_tracker) 
+ DeepSort [repo](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/deepsort_tracker) 
+ ByteTrack [repo](https://github.com/FoundationVision/ByteTrack/tree/main/yolox/tracker) 

# How to use
- `best_weight.pt`: Train a detection model for any object to get the best model file. Detection model [repo](https://www.kaggle.com/code/venon553/vehicletracking)
- `ckpt.t7`: Train a Reid model for any object to get the best model file. Reid model [repo](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6?fbclid=IwY2xjawGgP-JleHRuA2FlbQIxMAABHTtlXk48SZAIWwOCf8kLAtTkbbJEg0t3xx_npqBuIJl4xu0ZMND5xssGbQ_aem_AqvZbukQOnhFL1xYKFwPdA)

# Demo
```
pip install -r requirements.txt
python demo_ByteTrack.py
``` 
# Demo with streamlit application
```
pip install -r requirements.txt
python streamlit_app.py
```
# Evaluate
- Convert.py: convert the annotations from the annotation tool to the same format obtained from the file model --> Obtain the anno_gt_videox file (where x is the video status)
- run_ByteTrack.py: use the model to track --> Obtain the anno_pred_videox file (where x is the video status)
- evaluate_ByteTrack.py: conduct evaluation according to the metrics: MOTA, MOTP, IDF1.