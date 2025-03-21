git clone https://github.com/ultralytics/yolov5.git

# trained on first page of i9 2023
cat model_weights/i9_2016/1/detect_fields/best_chunk_* > model_weights/i9_2016/1/detect_fields/best.pt
cat model_weights/i9_2016/1/detect_form/best_chunk_* > model_weights/i9_2016/1/detect_form/best.pt
cat model_weights/i9_2016/2/detect_form/best_chunk_* > model_weights/i9_2016/2/detect_form/best.pt

# trained on first page of i9 2016
cat model_weights/i9_2016/2/detect_fields/best_chunk_* > model_weights/i9_2016/2/detect_fields/best.pt