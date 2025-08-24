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
import csv
from datetime import datetime

# ----------------------------
# 1. Performance Measurement Setup
# ----------------------------
# Create results directory
results_dir = "performance_results"
os.makedirs(results_dir, exist_ok=True)

# Create CSV file for results
csv_filename = f"{results_dir}/performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Initialize performance metrics
performance_metrics = {
    "frame_count": 0,
    "start_time": time.time(),
    "yolo_total_time": 0,
    "midas_total_time": 0,
    "total_processing_time": 0,
    "fps_history": [],
    "detection_counts": [],
    "depth_values": [],
    "tts_latencies": []
}

# Write CSV header
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'timestamp', 'frame_number', 'yolo_time', 'midas_time', 
        'total_processing_time', 'fps', 'object_count', 
        'avg_depth', 'min_depth', 'max_depth', 'tts_latency'
    ])

# ----------------------------
# 2. Display Configuration
# ----------------------------
MAX_DISPLAY_WIDTH = 1280  # Maximum width for display
MAX_DISPLAY_HEIGHT = 720  # Maximum height for display
LABEL_SCALE = 0.4         # Scale factor for labels
BOX_THICKNESS = 1         # Thickness of bounding boxes

# ----------------------------
# 3. Load Models
# ----------------------------
yolo_model_path = "models/yolo11s.pt"
yolo_model = YOLO(yolo_model_path)
print("YOLO11s PyTorch model loaded!")

midas_model_path = "models/midas.tflite"
midas = tf.lite.Interpreter(model_path=midas_model_path, num_threads=4)
midas.allocate_tensors()
input_details = midas.get_input_details()
output_details = midas.get_output_details()
midas_h, midas_w = input_details[0]['shape'][1:3]
print(f"MiDaS TFLite loaded. Input size: {midas_w}x{midas_h}")

# ----------------------------
# 4. TTS Setup
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
    
    if not text or text == last_spoken_text:
        return
    
    last_spoken_text = text
    
    # Measure TTS latency
    tts_start_time = time.time()
    
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
                
                # Record TTS latency
                tts_latency = time.time() - tts_start_time
                performance_metrics["tts_latencies"].append(tts_latency)
                
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                tts_busy = False
    
    thread = threading.Thread(target=tts_thread)
    thread.daemon = True
    thread.start()

# ----------------------------
# 5. Video Capture Setup
# ----------------------------
cap = cv2.VideoCapture(0)  # Use default USB camera (index 0)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    exit()

# Set camera resolution (optional, adjust based on your camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("USB camera initialized for video capture")

# ----------------------------
# 6. Mode Management
# ----------------------------
class SystemMode:
    ALERT_MODE = 1        # Mode 1: Only warns about close objects
    DESCRIPTION_MODE = 2  # Mode 2: Describes all objects

current_mode = SystemMode.ALERT_MODE

# ----------------------------
# 7. Angular Positioning Configuration
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
# 8. Image Resizing Functions
# ----------------------------
def resize_image_to_fit(image):
    """Resize image to fit within maximum display dimensions while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factors
    scale_width = MAX_DISPLAY_WIDTH / width
    scale_height = MAX_DISPLAY_HEIGHT / height
    scale = min(scale_width, scale_height, 1.0)  # Don't scale up, only down
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def calculate_scale_factor(original_width, original_height):
    """Calculate the scale factor used for resizing"""
    scale_width = MAX_DISPLAY_WIDTH / original_width
    scale_height = MAX_DISPLAY_HEIGHT / original_height
    return min(scale_width, scale_height, 1.0)

# ----------------------------
# 9. Object Tracker
# ----------------------------
class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.next_id = 0
        self.history_length = 15
        
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
            'depth_history': deque(maxlen=self.history_length),
            'last_center': (center_x, center_y),
            'class_id': class_id,
            'last_seen': time.time(),
            'stable_label': None,
            'label_count': 0
        }
        self.next_id += 1
        return obj_id
    
    def update_depth(self, obj_id, depth_value):
        if obj_id not in self.objects:
            return depth_value
            
        self.objects[obj_id]['depth_history'].append(depth_value)
        history = list(self.objects[obj_id]['depth_history'])
        if len(history) < 3:
            return depth_value
            
        weights = np.linspace(0.5, 1.5, len(history))
        weights = weights / weights.sum()
        return np.average(history, weights=weights)
    
    def get_stable_label(self, obj_id, current_label):
        if obj_id not in self.objects:
            return current_label
            
        obj = self.objects[obj_id]
        
        if obj['stable_label'] is None:
            obj['stable_label'] = current_label
            obj['label_count'] = 1
            return current_label
        
        if obj['stable_label'] == current_label:
            obj['label_count'] += 1
        else:
            obj['label_count'] -= 1
            
        if obj['label_count'] <= 0:
            obj['stable_label'] = current_label
            obj['label_count'] = 1
        elif obj['label_count'] > 8:
            obj['label_count'] = 8
            
        return obj['stable_label']
    
    def cleanup_old_objects(self):
        current_time = time.time()
        self.objects = {k: v for k, v in self.objects.items() if current_time - v['last_seen'] < 1.0}

tracker = ObjectTracker()

# ----------------------------
# 10. Performance Measurement Functions
# ----------------------------
def record_performance_metrics(yolo_time, midas_time, object_count, depth_values):
    """Record performance metrics for current frame"""
    performance_metrics["frame_count"] += 1
    performance_metrics["yolo_total_time"] += yolo_time
    performance_metrics["midas_total_time"] += midas_time
    
    total_time = yolo_time + midas_time
    performance_metrics["total_processing_time"] += total_time
    
    # Calculate current FPS
    if total_time > 0:
        current_fps = 1.0 / total_time
        performance_metrics["fps_history"].append(current_fps)
    
    # Record detection count
    performance_metrics["detection_counts"].append(object_count)
    
    # Record depth values
    if depth_values:
        performance_metrics["depth_values"].extend(depth_values)
    
    # Calculate averages for display
    avg_fps = np.mean(performance_metrics["fps_history"][-30:]) if performance_metrics["fps_history"] else 0
    avg_yolo = performance_metrics["yolo_total_time"] / performance_metrics["frame_count"] * 1000
    avg_midas = performance_metrics["midas_total_time"] / performance_metrics["frame_count"] * 1000
    avg_objects = np.mean(performance_metrics["detection_counts"][-30:]) if performance_metrics["detection_counts"] else 0
    
    # Write to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
            performance_metrics["frame_count"],
            yolo_time * 1000,  # Convert to ms
            midas_time * 1000,  # Convert to ms
            total_time * 1000,  # Convert to ms
            current_fps if total_time > 0 else 0,
            object_count,
            np.mean(depth_values) if depth_values else 0,
            np.min(depth_values) if depth_values else 0,
            np.max(depth_values) if depth_values else 0,
            performance_metrics["tts_latencies"][-1] if performance_metrics["tts_latencies"] else 0
        ])
    
    return avg_fps, avg_yolo, avg_midas, avg_objects

def print_performance_summary():
    """Print final performance summary"""
    total_time = time.time() - performance_metrics["start_time"]
    avg_fps = performance_metrics["frame_count"] / total_time
    
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total frames processed: {performance_metrics['frame_count']}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average YOLO time: {performance_metrics['yolo_total_time']/performance_metrics['frame_count']*1000:.2f} ms")
    print(f"Average MiDaS time: {performance_metrics['midas_total_time']/performance_metrics['frame_count']*1000:.2f} ms")
    
    if performance_metrics["detection_counts"]:
        print(f"Average objects per frame: {np.mean(performance_metrics['detection_counts']):.2f}")
    
    if performance_metrics["tts_latencies"]:
        print(f"Average TTS latency: {np.mean(performance_metrics['tts_latencies'])*1000:.2f} ms")
        print(f"TTS utterances: {len(performance_metrics['tts_latencies'])}")
    
    print(f"Results saved to: {csv_filename}")
    print("="*50)

# ----------------------------
# 11. Helper Functions
# ----------------------------
def run_midas(frame):
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
        return "VERY CLOSE", (0, 0, 255), True
    elif depth_value > 0.55:
        return "CLOSE", (0, 165, 255), True
    elif depth_value > 0.3:
        return "MEDIUM", (0, 255, 0), True
    else:
        return "FAR", (255, 255, 255), False

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

def save_result_frame(frame, output_folder="results"):
    """Save the processed video frame with detections"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"result_frame_{timestamp}.jpg")
    
    result_frame = frame.copy()
    height, width = result_frame.shape[:2]
    
    results = yolo_model(result_frame, imgsz=640, verbose=False, conf=0.6)
    depth_map = run_midas(result_frame)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            if x2 <= x1 or y2 <= y1 or score < 0.6:
                continue
            
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
                
            current_depth = np.median(depth_roi)
            current_distance, color, is_close = depth_label(current_depth)
            
            object_name = yolo_model.names[int(class_id)]
            x_center = (x1 + x2) // 2
            angle = calculate_angle(x_center, width)
            angular_pos = get_angular_position(angle)
            
            label = f"{object_name}: {current_distance} ({angular_pos})"
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS * 2)
            
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.5, BOX_THICKNESS * 2
            )
            cv2.rectangle(result_frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.5, (0, 0, 0), BOX_THICKNESS * 2)
    
    cv2.imwrite(output_path, result_frame)
    print(f"Result saved to: {output_path}")

# ----------------------------
# 12. Main Processing Loop
# ----------------------------
print("Video Testing Mode with Angular Positioning Active!")
print("Performance metrics will be recorded to CSV file")
print("Display Resolution: 1280x720 (frames will be scaled to fit)")
print("Angular Range: ±45° (straight ahead: ±5°)")
print("\nControls:")
print("'m' - Switch between Alert Mode and Description Mode")
print("'d' - Describe current frame")
print("'b' - Switch TTS language (English/Bangla)")
print("'s' - Toggle TTS on/off")
print("'w' - Save current frame")
print("'p' - Print performance summary")
print("'q' - Quit")

try:
    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera!")
            break
        
        # Start timing for performance measurement
        frame_start_time = time.time()
        
        # Resize for display
        display_frame = resize_image_to_fit(original_frame)
        original_height, original_width = original_frame.shape[:2]
        display_height, display_width = display_frame.shape[:2]
        
        # Calculate scale factor for coordinate conversion
        scale_x = display_width / original_width
        scale_y = display_height / original_height
        
        # Run YOLO model with timing
        yolo_start_time = time.time()
        results = yolo_model(original_frame, imgsz=640, verbose=False, conf=0.6)
        yolo_time = time.time() - yolo_start_time
        
        # Run MiDaS model with timing
        midas_start_time = time.time()
        depth_map = run_midas(original_frame)
        midas_time = time.time() - midas_start_time

        close_objects_info = []  # For Alert Mode: (name, distance, angle)
        all_objects_info = []    # For Description Mode: (name, distance, angle)
        depth_values = []        # For performance metrics

        object_count = 0
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                
                if x2 <= x1 or y2 <= y1 or score < 0.6:
                    continue
                
                depth_roi = depth_map[y1:y2, x1:x2]
                if depth_roi.size == 0:
                    continue
                    
                current_depth = np.median(depth_roi)
                depth_values.append(current_depth)
                current_distance, color, is_close = depth_label(current_depth)
                
                object_name = yolo_model.names[int(class_id)]
                x_center = (x1 + x2) // 2
                angle = calculate_angle(x_center, original_width)
                angular_pos = get_angular_position(angle)
                
                # Scale coordinates for display
                display_x1 = int(x1 * scale_x)
                display_y1 = int(y1 * scale_y)
                display_x2 = int(x2 * scale_x)
                display_y2 = int(y2 * scale_y)
                
                label = f"{object_name}: {current_distance} ({angular_pos})"
                
                all_objects_info.append((object_name, current_distance, angle))
                if is_close:
                    close_objects_info.append((object_name, current_distance, angle))
                
                object_count += 1
                
                # Draw on display frame
                cv2.rectangle(display_frame, (display_x1, display_y1), (display_x2, display_y2), color, BOX_THICKNESS)
                
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, BOX_THICKNESS
                )
                cv2.rectangle(display_frame, (display_x1, display_y1 - label_height - 10), 
                             (display_x1 + label_width, display_y1), color, -1)
                cv2.putText(display_frame, label, (display_x1, display_y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 0, 0), BOX_THICKNESS)

        # Record performance metrics
        avg_fps, avg_yolo, avg_midas, avg_objects = record_performance_metrics(
            yolo_time, midas_time, object_count, depth_values
        )

        # Generate messages based on current mode
        current_time = time.time()
        if current_mode == SystemMode.ALERT_MODE:
            alert_message = generate_alert_message(close_objects_info)
            description_message = alert_message
            if alert_message and current_time - last_alert_time > alert_cooldown and not tts_busy:
                speak_text(alert_message, current_tts_language)
                last_alert_time = current_time
        else:
            alert_message = None
            description_message = describe_frame(all_objects_info)
        
        # Display info
        cv2.putText(display_frame, "USB Camera Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        mode_text = "ALERT MODE" if current_mode == SystemMode.ALERT_MODE else "DESCRIPTION MODE"
        cv2.putText(display_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        lang_text = f"TTS: {current_tts_language.upper()} {'ON' if tts_enabled else 'OFF'}"
        cv2.putText(display_frame, lang_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Display performance metrics
        perf_text = f"FPS: {avg_fps:.1f} | YOLO: {avg_yolo:.1f}ms | MiDaS: {avg_midas:.1f}ms | Objects: {avg_objects:.1f}"
        cv2.putText(display_frame, perf_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        if description_message:
            # Split long message into multiple lines
            words = description_message.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 60:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            for i, line in enumerate(lines):
                y_pos = display_frame.shape[0] - 60 - (len(lines) - i - 1) * 25
                color = (0, 0, 255) if current_mode == SystemMode.ALERT_MODE else (255, 255, 255)
                cv2.putText(display_frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        controls = "Controls: m=Mode d=Describe b=Lang s=TTS w=Save p=Perf q=Quit"
        cv2.putText(display_frame, controls, (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Angular Object Detection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF  # Process frames continuously, check for key press
        
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_mode = SystemMode.DESCRIPTION_MODE if current_mode == SystemMode.ALERT_MODE else SystemMode.ALERT_MODE
            mode_name = "Description Mode" if current_mode == SystemMode.DESCRIPTION_MODE else "Alert Mode"
            print(f"Switched to {mode_name}")
            speak_text(f"Switched to {mode_name}", current_tts_language)
        elif key == ord('d'):
            print(f"OUTPUT: {description_message}")
            speak_text(description_message, current_tts_language)
        elif key == ord('b'):
            current_tts_language = "bangla" if current_tts_language == "english" else "english"
            print(f"TTS language switched to {current_tts_language}")
            speak_text(f"Language switched to {current_tts_language}", current_tts_language)
        elif key == ord('s'):
            tts_enabled = not tts_enabled
            status = "enabled" if tts_enabled else "disabled"
            print(f"TTS {status}")
            speak_text(f"TTS {status}", current_tts_language)
        elif key == ord('w'):
            save_result_frame(original_frame)
            print("Frame saved with high-quality labels!")
        elif key == ord('p'):
            print_performance_summary()
        
        # Clean up old objects to handle dynamic video content
        tracker.cleanup_old_objects()

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print_performance_summary()
    print("Video testing completed!")