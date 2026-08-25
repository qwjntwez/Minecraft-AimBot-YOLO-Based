# Minecraft Aim Bot — Neural Network Based (YOLO26)

> 🚧 **Work in Progress**

A computer vision-based aim bot for Minecraft using a custom-trained YOLO26 model for real-time player detection.

## Current Features

- Real-time screen capture.
- YOLO26 inference pipeline:
  - image preprocessing;
  - player detection;
  - bounding box coordinate transformation from YOLO26 input resolution back to the original screen resolution.
- Target tracking algorithm for smoother and more stable aiming.
- Automatic aiming at detected targets.
- Multithreaded pipeline — screen capture, model inference, and aiming run in separate threads.

## Planned Features

- Automatic attack system with cooldown adjustment based on the currently equipped weapon.
- GUI.

-----------------------------------------------------------------

### Model Configuration

- **Training dataset:** https://shorturl.at/GvnOI
- **Pretrained model:** `model/model.onnx`
  
- Architecture: YOLO26n
- Input size: 640×640
- Format: ONNX
- Task: Object Detection
- Classes: 1 (`player`)

### Dataset Split

| Split | Percentage |
| --- | ---: |
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Test Metrics

| Metric | Score |
| --- | ---: |
| Precision | 0.772 |
| Recall | 0.735 |
| mAP@50 | 0.824 |
| mAP@50–95 | 0.511 |
