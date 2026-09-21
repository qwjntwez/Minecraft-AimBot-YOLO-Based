# Minecraft Aim Bot — Neural Network Based (YOLO26)

> 🚧 **Work in Progress**

A computer vision-based aim bot for Minecraft using a custom-trained YOLO26 model for real-time player detection.

## Current Features

- Real-time screen capture.
- YOLO26 inference pipeline:
  - image preprocessing;
  - player detection;
  - bounding box coordinate transformation from YOLO26 input resolution back to the original screen resolution.
- OC-SORT tracking algorithm for smoother and more stable aiming.
- Automatic aiming at detected targets.
- Multithreaded pipeline — screen capture, model inference, and aiming run in separate threads.
- Automatic attack system with cooldown adjustment based on the currently equipped weapon.

## Planned Features
- Performance optimization.
- GUI.

## Auto Attack
- Auto-attack is implemented using a game mod that transmits the player's state via UDP, including whether the target is within attack range and the current attack cooldown.

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
| FP32 |
| Metric | Score |
| --- | ---: |
| Precision | 0.915 |
| Recall | 0.792 |
| mAP@50 | 0.89 |
| mAP@50–95 | 0.58 |

| FP16 |
| Metric | Score |
| --- | ---: |
| Precision | 0.928 |
| Recall | 0.781 |
| mAP@50 | 0.888 |
| mAP@50–95 | 0.588 |
