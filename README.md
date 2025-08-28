## KHVR Lightweight Framework for Edge Device
This Lightweight Framework is designed as a nevigation system for robot and to assist visually impaired users by providing real-time audio alerts and descriptions of their surroundings. It can be used as a guidance tool, offering features like object detection, distance estimation, and multilingual audio feedback (English and Bangla). The framework leverages computer vision, machine learning, and text-to-speech technologies, optimized for lightweight edge devices.
Overview

## Purpose: Robotics vision and situational awareness for visually impaired users.
Modes:
Alert Mode: Warns about nearby hazards (e.g., "Warning: a car straight ahead is too close!").
Description Mode: Describes all visible objects (e.g., "car straight ahead, and 2 dogs 30 degree left").


Languages: Audio feedback in English or Bangla (object names in English), with English on-screen text.
Controls: Toggle modes, languages, audio, and save snapshots using keyboard commands.

## Prerequisites

Hardware: (e.g.any local machine, Raspberry Pi).
Camera: USB webcam or compatible Camera Module.
Internet Connection: Required for downloading dependencies and model files.
Storage: At least 10GB free space.
Power Supply: Stable 5V/3A power supply.

## Installation
Step 1: Clone the Repository
Clone this repository to your local machine:
git clone https://github.com/iftitalukder/RoboVision_Edge_lightweight.git
cd RoboVision_Edge_lightweight

Step 2: Update System
Ensure your system is up-to-date:
sudo apt update && sudo apt upgrade -y

Step 3: Install System Dependencies
Install required libraries for OpenCV, Pygame, and numerical operations:
sudo apt install -y python3 python3-pip libatlas-base-dev libopenjp2-7 libtiff5 libjpeg-dev libavcodec-dev libavformat-dev libswscale-dev libsdl2-dev libsdl2-mixer-dev g++ cmake

Step 4: Set Up Python Environment
Verify Python version (requires 3.10):
python3 --version

If needed, install Python 3.10:
sudo apt install -y python3.10

Upgrade pip:
pip3 install --upgrade pip

Step 5: Install Python Dependencies
Install dependencies from requirements.txt:
pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

Note: The --index-url ensures CPU-only PyTorch wheels, suitable for edge devices.
Optional: Optimized TensorFlow
For better performance on raspberry pi, install an ARM-optimized TensorFlow wheel:
pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.11.0/tensorflow-2.11.0-cp39-none-linux_aarch64.whl

This replaces the default TensorFlow from requirements.txt.
Step 6: Set Up Camera
Check if the camera is detected:
ls /dev/video*

For a compatible Camera Module, enable it:
sudo raspi-config

Select "Interface Options" > "Camera" > Enable. Reboot if required:
sudo reboot

Step 7: Download Model Files
Download required models:

YOLOv11 Model (yolo11s.pt): Ultralytics YOLOv11 releases.
MiDaS TFLite Model (midas.tflite): Intel MiDaS repository.

Create a models/ directory and place the files:
mkdir models

Move yolo11s.pt and midas.tflite to the models/ directory.

Step 8: Increase Swap (Optional)
To avoid memory issues, increase swap space:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

Step 9: Verify Installation
Create a test script (check_deps.py):

import cv2
import torch
import numpy
import tensorflow
from ultralytics import YOLO
from gtts import gTTS
import pygame
from googletrans import Translator
print("All dependencies imported successfully!")

python3 check_deps.py

Resolve any errors by revisiting the relevant step.

## Run it:

Execute the main script:

python khvr.py (optimized for edge devices)


Controls will be avaible on terminal while running script

## Learning About the Framework

Read my paper to learn in depth about the framework

## Troubleshooting

Camera Not Detected: Check /dev/video* and enable the camera in raspi-config.
Memory Errors: Increase swap or close applications.
TensorFlow Issues: Use the ARM-optimized wheel or simplify the model.
OpenCV GUI Issues: Install python3-opencv:sudo apt install -y python3-opencv


## Slow Performance: Reduce resolution or frame rate in the script.

Notes

Ensure yolo11s.pt and midas.tflite are in the models/ directory.
OpenCV may not render Bangla text well; English is used for on-screen display.
The googletrans library requires an internet connection.

Contributing
## Feel free to fork this repository, make improvements, and submit pull requests! if you use it for research purpoose cite this respository.
