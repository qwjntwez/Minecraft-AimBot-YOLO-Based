# Minecraft Aim Bot — Neural Network Based (YOLO26)

> 🚧 **Work in Progress**

A computer vision-based aim bot for Minecraft using a custom-trained YOLO26 model for real-time player detection.

## Current Features

- Real-time screen capture.
- YOLO26 inference pipeline:
  - image preprocessing;
  - player detection;
  - bounding box coordinate transformation from YOLO26 input resolution back to the original screen resolution.
- Automatic aiming at detected targets.
- Multithreaded pipeline — screen capture, model inference, and aiming run in separate threads.

## Planned Features

- Automatic attack system with cooldown adjustment based on the currently equipped weapon.
- Target tracking algorithm for smoother and more stable aiming.
- GUI.
