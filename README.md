# Blind_Assist
# This blind assist system helps visually impaired users navigate safely by providing real-time audio alerts and descriptions of their surroundings. In Alert Mode, it warns about nearby objects (e.g., "Warning: a car straight ahead is too close!") every 5 seconds, ensuring timely hazard notifications. In Description Mode, it describes all visible objects (e.g., "car straight ahead, and 2 dogs 30 degree left") with the same frequency, aiding environmental awareness. Audio feedback is available in English or Bangla (keeping object names in English for clarity), while on-screen text remains in English for sighted assistants. Users can switch modes, toggle between languages, enable/disable audio, or save snapshots using simple keyboard commands, making it a practical tool for safe navigation and situational understanding.
# Setup Guide for Object Detection Script on Raspberry Pi 4 (Xubuntu)

This guide explains how to set up the environment to run the object detection script on a Raspberry Pi 4 (8GB) running Xubuntu. The script uses computer vision, machine learning, and text-to-speech for real-time object detection with Bangla/English TTS support.

## Prerequisites
- **Hardware**: Raspberry Pi 4 (8GB) with Xubuntu installed.
- **Camera**: USB webcam or Raspberry Pi Camera Module.
- **Internet Connection**: Required for downloading dependencies and model files.
- **Storage**: At least 10GB free space for dependencies and models.
- **Power Supply**: Stable 5V/3A power supply to avoid performance issues.

## Step 1: Update System
Ensure your system is up-to-date:
```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Install System Dependencies
Install required system libraries for OpenCV, Pygame, and numerical operations:
```bash
sudo apt install -y python3 python3-pip libatlas-base-dev libopenjp2-7 libtiff5 libjpeg-dev libavcodec-dev libavformat-dev libswscale-dev libsdl2-dev libsdl2-mixer-dev g++ cmake
```

## Step 3: Set Up Python Environment
The script requires Python 3.8 or higher. Xubuntu typically includes Python 3. Verify:
```bash
python3 --version
```
If Python 3.8+ is not installed, install it:
```bash
sudo apt install -y python3.9
```
Upgrade pip:
```bash
pip3 install --upgrade pip
```

## Step 4: Install Python Dependencies
Clone or download the repository containing the script and `requirements.txt`. Navigate to the project directory:
```bash
cd /path/to/your/repository
```
Install Python dependencies:
```bash
pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```
**Note**: The `--index-url` ensures PyTorch uses CPU-only wheels, suitable for Raspberry Pi.

### Optional: Optimized TensorFlow
TensorFlow can be heavy. For better performance, install an ARM-optimized wheel:
```bash
pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.11.0/tensorflow-2.11.0-cp39-none-linux_aarch64.whl
```
This replaces the `tensorflow` installation from `requirements.txt`.

## Step 5: Set Up Camera
Ensure your camera is detected:
```bash
ls /dev/video*
```
For a Raspberry Pi Camera Module, enable it:
```bash
sudo raspi-config
```
Select "Interface Options" > "Camera" > Enable. Reboot if required:
```bash
sudo reboot
```

## Step 6: Download Model Files
The script requires two model files:
- **YOLOv11 Model** (`yolo11s.pt`): Download from the [Ultralytics YOLOv11 releases](https://github.com/ultralytics/ultralytics/releases).
- **MiDaS TFLite Model** (`midas.tflite`): Download from the [Intel MiDaS repository](https://github.com/isl-org/MiDaS).

Create a `models/` directory in the project root:
```bash
mkdir models
```
Place `yolo11s.pt` and `midas.tflite` in the `models/` directory.

## Step 7: Increase Swap (Optional)
To prevent memory issues during installation or runtime, increase swap space:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Step 8: Verify Installation
Create a test script (`check_deps.py`) to verify dependencies:
```python
import cv2
import torch
import numpy
import tensorflow
from ultralytics import YOLO
from gtts import gTTS
import pygame
from googletrans import Translator
print("All dependencies imported successfully!")
```
Run it:
```bash
python3 check_deps.py
```
If errors occur, revisit the relevant installation step.

## Step 9: Run the Script
Run the object detection script:
```bash
python3 object_detection.py
```
### Controls
- `t`: Toggle between Continuous Alerts and Frame Description modes.
- `n`: Describe the current frame (in Frame Description mode).
- `b`: Toggle TTS language (English/Bangla).
- `q`: Quit the program.

## Troubleshooting
- **Camera Not Detected**: Check `/dev/video*` and ensure the camera is enabled in `raspi-config`.
- **Memory Errors**: Increase swap size or close other applications.
- **TensorFlow Issues**: Use the ARM-optimized wheel or reduce model complexity.
- **OpenCV GUI Issues**: If `cv2.imshow` fails, ensure `python3-opencv` is installed:
  ```bash
  sudo apt install -y python3-opencv
  ```
- **Slow Performance**: Reduce YOLO image size (`imgsz=320`) in the script or use a lighter model.



## Notes
- **Model Files**: Ensure `yolo11s.pt` and `midas.tflite` are in the `models/` directory relative to the script.
- **Font Rendering**: OpenCV may not render Bangla text well. The script uses English for on-screen display, but if Bangla is needed, consider using `Pillow` for text rendering.
- **Internet**: The `googletrans` library requires an internet connection for translations.
