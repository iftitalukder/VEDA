import cv2
import torch
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import time
from gtts import gTTS
import pygame
import io
import threading
import os
import math
import re

# ----------------------------
# 1. Performance Optimization Settings
# ----------------------------
# Reduced resolution for Raspberry Pi
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_SKIP = 2  # Process every nth frame
frame_counter = 0

# Disable pygame video initialization to save resources
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.display.set_mode((1, 1))  # Minimal display setup

# ----------------------------
# 2. Load Models with Optimizations
# ----------------------------
# Set TensorFlow to use only one thread to avoid conflicts
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

yolo_model_path = "models/yolo11s.pt"
yolo_model = YOLO(yolo_model_path)
print("YOLO11s PyTorch model loaded!")

# Use smaller input size for MiDaS on Raspberry Pi
midas_model_path = "models/midas.tflite"
midas = tf.lite.Interpreter(model_path=midas_model_path, num_threads=2)  # Reduced threads
midas.allocate_tensors()
input_details = midas.get_input_details()
output_details = midas.get_output_details()
midas_h, midas_w = 256, 256  # Reduced input size for MiDaS
print(f"MiDaS TFLite loaded. Input size: {midas_w}x{midas_h}")

# ----------------------------
# 3. TTS Setup
# ----------------------------
pygame.mixer.init()
current_tts_language = "english"
tts_enabled = True
last_spoken_text = ""
tts_lock = threading.Lock()
tts_busy = False
last_alert_time = 0
alert_cooldown = 5.0  # Seconds between alerts

# Predefined English-to-Bangla translations for efficiency
TRANSLATIONS = {
    # Common phrases
    "Warning": "সতর্কতা",
    "is too close": "খুব কাছে",
    "are too close": "খুব কাছে",
    "No objects detected": "কোনো বস্তু সনাক্ত করা হয়নি",
    "Switched to Alert Mode": "সতর্ক মোডে স্যুইচ করা হয়েছে",
    "Switched to Description Mode": "বর্ণনা মোডে স্যুইচ করা হয়েছে",
    "TTS enabled": "টিটিএস সক্ষম করা হয়েছে",
    "TTS disabled": "টিটিএস নিষ্ক্রিয় করা হয়েছে",
    "Language switched to bangla": "ভাষা বাংলায় স্যুইচ করা হয়েছে",
    "Language switched to english": "ভাষা ইংরেজিতে স্যুইচ করা হয়েছে",
    # Distance labels
    "VERY CLOSE": "খুব কাছে",
    "CLOSE": "কাছে",
    "MEDIUM": "মাঝারি",
    "FAR": "দূরে",
    # Angular positions
    "straight ahead": "সোজা সামনে",
    "degree left": "ডিগ্রি বামে",
    "degree right": "ডিগ্রি ডানে",
}

def translate_to_bangla(text):
    """Translate English text to Bangla, keeping object names in English"""
    result = text
    # Replace known phrases
    for eng, ban in TRANSLATIONS.items():
        result = result.replace(eng, ban)
    
    # Handle numbers for degrees (e.g., "30 degree left" -> "30 ডিগ্রি বামে")
    result = re.sub(r'(\d+) degree left', r'\1 ডিগ্রি বামে', result)
    result = re.sub(r'(\d+) degree right', r'\1 ডিগ্রি ডানে', result)
    
    # Handle plural forms, keeping object names in English (e.g., "2 cars" -> "2 cars")
    result = re.sub(r'(\d+) (\w+)s', r'\1 \2s', result)
    
    # Remove indefinite article "a" for Bangla alerts
    result = result.replace("a ", "")
    
    return result

def speak_text(text, language="english"):
    """Convert text to speech in background thread without overlap"""
    global tts_busy, last_spoken_text
    
    if not text or text == last_spoken_text or not tts_enabled:
        return
    
    last_spoken_text = text
    
    def tts_thread():
        global tts_busy
        with tts_lock:
            tts_busy = True
            try:
                lang_code = "bn" if language == "bangla" else "en"
                text_to_speak = translate_to_bangla(text) if language == "bangla" else text
                tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                pygame.mixer.music.load(mp3_fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                tts_busy = False
    
    thread = threading.Thread(target=tts_thread)
    thread.daemon = True
    thread.start()

# ----------------------------
# 4. Video Capture Setup
# ----------------------------
cap = cv2.VideoCapture(0)  # Use default USB camera (index 0)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    exit()

# Set lower camera resolution for Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS
print(f"USB camera initialized at {CAMERA_WIDTH}x{CAMERA_HEIGHT}")

# ----------------------------
# 5. Mode Management
# ----------------------------
class SystemMode:
    ALERT_MODE = 1        # Mode 1: Only warns about close objects
    DESCRIPTION_MODE = 2  # Mode 2: Describes all objects

current_mode = SystemMode.ALERT_MODE

# ----------------------------
# 6. Angular Positioning Configuration
# ----------------------------
MAX_ANGLE = 45  # Maximum 45° left/right
STRAIGHT_THRESHOLD = 5  # ±5° is considered "straight ahead"

def calculate_angle(x_center, frame_width):
    """Calculate angle from center in degrees"""
    center_x = frame_width / 2
    offset = x_center - center_x
    normalized_offset = offset / center_x  # -1 to 1
    angle = normalized_offset * MAX_ANGLE
    return angle

def get_angular_position(angle):
    """Convert angle to human-readable position"""
    if abs(angle) <= STRAIGHT_THRESHOLD:
        return "straight ahead"
    elif angle < 0:
        return f"{abs(angle):.0f} degree left"
    else:
        return f"{angle:.0f} degree right"

# ----------------------------
# 7. Object Tracker (Simplified)
# ----------------------------
class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.next_id = 0
        
    def get_object_id(self, box, class_id):
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        for obj_id, obj_data in self.objects.items():
            prev_center_x, prev_center_y = obj_data['last_center']
            prev_class = obj_data['class_id']
            
            if (prev_class == class_id and 
                abs(center_x - prev_center_x) < 50 and 
                abs(center_y - prev_center_y) < 50):
                obj_data['last_center'] = (center_x, center_y)
                obj_data['last_seen'] = time.time()
                return obj_id
        
        obj_id = self.next_id
        self.objects[obj_id] = {
            'last_center': (center_x, center_y),
            'class_id': class_id,
            'last_seen': time.time(),
        }
        self.next_id += 1
        return obj_id
    
    def cleanup_old_objects(self):
        current_time = time.time()
        self.objects = {k: v for k, v in self.objects.items() if current_time - v['last_seen'] < 1.0}

tracker = ObjectTracker()

# ----------------------------
# 8. Helper Functions
# ----------------------------
def run_midas(frame):
    # Use smaller input for faster processing
    frame_resized = cv2.resize(frame, (midas_w, midas_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_data = np.expand_dims(frame_rgb, axis=0)
    midas.set_tensor(input_details[0]['index'], input_data)
    midas.invoke()
    depth_map = midas.get_tensor(output_details[0]['index'])[0, :, :, 0]
    
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_resized = cv2.resize(depth_normalized, (frame.shape[1], frame.shape[0]))
    return depth_resized

def depth_label(depth_value):
    if depth_value > 0.75:
        return "VERY CLOSE", True
    elif depth_value > 0.55:
        return "CLOSE", True
    elif depth_value > 0.3:
        return "MEDIUM", True
    else:
        return "FAR", False

def format_object_list(objects_info, for_alert=False):
    """Format objects with proper counting, grammar, and angular position"""
    object_groups = {}
    
    for obj_name, distance, angle in objects_info:
        if for_alert:
            if distance in ["VERY CLOSE", "CLOSE", "MEDIUM"]:
                position = get_angular_position(angle)
                if position not in object_groups:
                    object_groups[position] = {}
                if obj_name not in object_groups[position]:
                    object_groups[position][obj_name] = 0
                object_groups[position][obj_name] += 1
        else:
            position = get_angular_position(angle)
            if position not in object_groups:
                object_groups[position] = {}
            if obj_name not in object_groups[position]:
                object_groups[position][obj_name] = 0
            object_groups[position][obj_name] += 1
    
    if not object_groups:
        return None
    
    position_descriptions = []
    
    for position, objects in object_groups.items():
        object_list = []
        for obj_name, count in objects.items():
            if count == 1:
                object_list.append(f"a {obj_name}" if for_alert else f"{obj_name}")
            else:
                object_list.append(f"{count} {obj_name}s")
        
        if object_list:
            if len(object_list) == 1:
                objects_str = object_list[0]
            else:
                objects_str = " and ".join(object_list) if len(object_list) == 2 else ", ".join(object_list[:-1]) + f", and {object_list[-1]}"
            
            position_descriptions.append(f"{objects_str} {position}")
    
    return position_descriptions

def generate_alert_message(close_objects_info):
    """Generate alert for close objects with angular position"""
    position_descriptions = format_object_list(close_objects_info, for_alert=True)
    
    if not position_descriptions:
        return None
    
    if len(position_descriptions) == 1:
        return f"Warning: {position_descriptions[0]} is too close!"
    else:
        return f"Warning: {', '.join(position_descriptions)} are too close!"

def describe_frame(objects_info):
    """Generate detailed description of all objects with angular position, without 'There is/are'"""
    position_descriptions = format_object_list(objects_info, for_alert=False)
    
    if not position_descriptions:
        return "No objects detected."
    
    if len(position_descriptions) == 1:
        return position_descriptions[0]
    else:
        return ", ".join(position_descriptions[:-1]) + f", and {position_descriptions[-1]}"

# ----------------------------
# 9. Main Processing Loop
# ----------------------------
print("Headless Angular Object Detection Active!")
print(f"Camera Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print("Angular Range: ±45° (straight ahead: ±5°)")
print("\nControls via terminal input:")
print("'m' - Switch between Alert Mode and Description Mode")
print("'d' - Describe current frame")
print("'b' - Switch TTS language (English/Bangla)")
print("'s' - Toggle TTS on/off")
print("'q' - Quit")

# Function to check for keyboard input without GUI
def check_keyboard_input():
    # This is a simple non-blocking input check for Raspberry Pi
    # For a more robust solution, consider using libraries like pynput or evdev
    try:
        import sys
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
    except:
        pass
    return None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera!")
            break
        
        # Skip frames to improve performance
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue
            
        # Run models on frame
        results = yolo_model(frame, imgsz=320, verbose=False, conf=0.5)  # Reduced size and confidence
        depth_map = run_midas(frame)

        close_objects_info = []  # For Alert Mode: (name, distance, angle)
        all_objects_info = []    # For Description Mode: (name, distance, angle)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                if x2 <= x1 or y2 <= y1 or score < 0.5:  # Reduced confidence threshold
                    continue
                
                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                    
                current_depth = np.median(depth_roi)
                current_distance, is_close = depth_label(current_depth)
                
                object_name = yolo_model.names[int(class_id)]
                x_center = (x1 + x2) // 2
                angle = calculate_angle(x_center, frame.shape[1])
                
                all_objects_info.append((object_name, current_distance, angle))
                if is_close:
                    close_objects_info.append((object_name, current_distance, angle))

        # Generate messages based on current mode
        current_time = time.time()
        if current_mode == SystemMode.ALERT_MODE:
            alert_message = generate_alert_message(close_objects_info)
            description_message = alert_message
            if alert_message and current_time - last_alert_time > alert_cooldown and not tts_busy:
                print(f"ALERT: {alert_message}")
                speak_text(alert_message, current_tts_language)
                last_alert_time = current_time
        else:
            alert_message = None
            description_message = describe_frame(all_objects_info)
            if description_message and description_message != last_spoken_text:
                print(f"DESCRIPTION: {description_message}")
                speak_text(description_message, current_tts_language)
        
        # Check for keyboard input
        key = check_keyboard_input()
        
        if key == 'q':
            break
        elif key == 'm':
            current_mode = SystemMode.DESCRIPTION_MODE if current_mode == SystemMode.ALERT_MODE else SystemMode.ALERT_MODE
            mode_name = "Description Mode" if current_mode == SystemMode.DESCRIPTION_MODE else "Alert Mode"
            print(f"Switched to {mode_name}")
            speak_text(f"Switched to {mode_name}", current_tts_language)
        elif key == 'd':
            description_message = describe_frame(all_objects_info)
            print(f"DESCRIPTION: {description_message}")
            speak_text(description_message, current_tts_language)
        elif key == 'b':
            current_tts_language = "bangla" if current_tts_language == "english" else "english"
            print(f"TTS language switched to {current_tts_language}")
            speak_text(f"Language switched to {current_tts_language}", current_tts_language)
        elif key == 's':
            tts_enabled = not tts_enabled
            status = "enabled" if tts_enabled else "disabled"
            print(f"TTS {status}")
            speak_text(f"TTS {status}", current_tts_language)
        
        # Clean up old objects to handle dynamic video content
        tracker.cleanup_old_objects()
        
        # Small delay to reduce CPU usage
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down...")

finally:
    cap.release()
    print("Video processing completed!")