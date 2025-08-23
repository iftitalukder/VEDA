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
from googletrans import Translator

# ----------------------------
# 1. Load Models
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
# 2. Translation Setup
# ----------------------------
translator = Translator()

# Dictionary for common object names (English to Bangla)
object_name_translations = {
    "person": "ব্যক্তি",
    "car": "গাড়ি",
    "dog": "কুকুর",
    "cat": "বিড়াল",
    "chair": "চেয়ার",
    # Add more translations as needed for YOLO model classes
}

# Dictionary for distance labels
distance_translations = {
    "VERY CLOSE": "খুব কাছাকাছি",
    "CLOSE": "কাছাকাছি",
    "MEDIUM": "মাঝারি",
    "FAR": "দূরে"
}

# Dictionary for section translations
section_translations = {
    "in front": "সামনে",
    "on the left": "বাম দিকে",
    "on the right": "ডান দিকে"
}

# Dictionary for other phrases
phrase_translations = {
    "is too close!": "খুব কাছাকাছি!",
    "are too close!": "খুব কাছাকাছি!",
    "No objects detected.": "কোন বস্তু সনাক্ত করা হয়নি।",
    "Language switched to english": "ভাষা ইংরেজিতে পরিবর্তিত হয়েছে",
    "Language switched to bangla": "ভাষা বাংলায় পরিবর্তিত হয়েছে",
    "Switched to Continuous Alerts mode": "নিরবচ্ছিন্ন সতর্কতা মোডে পরিবর্তিত হয়েছে",
    "Switched to Frame Description mode": "ফ্রেম বর্ণনা মোডে পরিবর্তিত হয়েছে",
    "Close objects": "কাছাকাছি বস্তু",
    "Press 'n' to describe frame": "ফ্রেম বর্ণনা করতে 'n' চাপুন",
    "Controls: 't'=Toggle Mode | 'n'=Describe Frame | 'b'=Toggle Language | 'q'=Quit": "নিয়ন্ত্রণ: 't'=মোড টগল | 'n'=ফ্রেম বর্ণনা | 'b'=ভাষা টগল | 'q'=বন্ধ",
    "TTS Language": "টিটিএস ভাষা",
    "mode": "মোড"
}

def translate_text(text, lang="en", for_tts=False):
    """Translate text to Bangla with natural phrasing for TTS/terminal"""
    if lang == "bn" and for_tts:
        # Check if text is in predefined translations
        if text in phrase_translations:
            return phrase_translations[text]
        if text in distance_translations:
            return distance_translations[text]
        if text in section_translations:
            return section_translations[text]
        if text.lower() in object_name_translations:
            return object_name_translations[text.lower()]
        # Fallback to googletrans
        try:
            return translator.translate(text, dest="bn").text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    return text

# Flag to control terminal output language (set to False for English terminal)
TERMINAL_IN_ENGLISH = True

# ----------------------------
# 3. TTS Setup
# ----------------------------
pygame.mixer.init()
current_tts_language = "english"  # Default language
tts_lock = threading.Lock()
tts_busy = False

def speak_text(text, language="english"):
    """Convert text to speech in background thread without overlap"""
    global tts_busy

    def tts_thread():
        global tts_busy
        with tts_lock:
            try:
                tts_busy = True
                lang_code = "bn" if language == "bangla" else "en"
                translated_text = translate_text(text, lang_code, for_tts=True)
                tts = gTTS(text=translated_text, lang=lang_code, slow=False)
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
    
    if not tts_busy:
        thread = threading.Thread(target=tts_thread)
        thread.daemon = True
        thread.start()

# ----------------------------
# 4. Camera setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# ----------------------------
# 5. Mode Management
# ----------------------------
class SystemMode:
    CONTINUOUS_ALERT = 1  # Mode 1: Continuous alerts for close objects
    FRAME_DESCRIPTION = 2 # Mode 2: Describe specific frame

current_mode = SystemMode.CONTINUOUS_ALERT
last_alert_time = 0
alert_cooldown = 5.0  # Seconds between alerts to allow TTS completion

# ----------------------------
# 6. Object Tracker
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
    
    def cleanup_old_objects(self, max_age=2.0):
        current_time = time.time()
        ids_to_remove = []
        for obj_id, obj_data in self.objects.items():
            if current_time - obj_data['last_seen'] > max_age:
                ids_to_remove.append(obj_id)
        for obj_id in ids_to_remove:
            del self.objects[obj_id]

tracker = ObjectTracker()

# ----------------------------
# 7. Helper Functions
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

def get_section(x_center, frame_width):
    """Determine if object is on left, middle, or right"""
    third = frame_width / 3
    if x_center < third:
        return "left"
    elif x_center < 2 * third:
        return "middle"
    else:
        return "right"

def format_object_list(objects_info, for_alert=False):
    """Format objects with proper counting, grammar, and section info"""
    object_counts = {}
    section_counts = {"left": {}, "middle": {}, "right": {}}
    
    for obj_name, distance, section in objects_info:
        # Translate for TTS if in Bangla mode
        tts_obj_name = translate_text(obj_name, "bn" if current_tts_language == "bangla" else "en", for_tts=True)
        tts_distance = translate_text(distance, "bn" if current_tts_language == "bangla" else "en", for_tts=True)
        
        if for_alert:
            # For alerts, only include if distance indicates closeness
            if distance in ["VERY CLOSE", "CLOSE", "MEDIUM"]:
                object_counts[tts_obj_name] = object_counts.get(tts_obj_name, 0) + 1
                if tts_obj_name not in section_counts[section]:
                    section_counts[section][tts_obj_name] = 0
                section_counts[section][tts_obj_name] += 1
        else:
            # For description, include all objects
            object_counts[tts_obj_name] = object_counts.get(tts_obj_name, 0) + 1
            if tts_obj_name not in section_counts[section]:
                section_counts[section][tts_obj_name] = 0
            section_counts[section][tts_obj_name] += 1
    
    if not object_counts:
        return None, None
    
    # Build formatted list with section info
    formatted_objects = []
    section_descriptions = []
    
    for section in ["left", "middle", "right"]:
        if section_counts[section]:
            section_objects = []
            for obj_name, count in section_counts[section].items():
                if current_tts_language == "bangla":
                    if count == 1:
                        section_objects.append(f"একটি {obj_name}")
                    else:
                        section_objects.append(f"{count} {obj_name}")
                else:
                    if count == 1:
                        section_objects.append(f"a {obj_name}")
                    else:
                        section_objects.append(f"{count} {obj_name}s")
            
            if section_objects:
                section_str = " and ".join(section_objects) if len(section_objects) == 2 else ", ".join(section_objects[:-1]) + f", and {section_objects[-1]}"
                
                section_text = "in front" if section == "middle" else f"on the {section}"
                if current_tts_language == "bangla":
                    section_text = translate_text(section_text, "bn", for_tts=True)
                
                section_descriptions.append(f"{section_str} {section_text}")
    
    return formatted_objects, section_descriptions

def generate_alert_message(close_objects_info):
    """Generate alert for close objects with natural phrasing"""
    formatted_objects, section_descriptions = format_object_list(close_objects_info, for_alert=True)
    
    if not section_descriptions:
        return None
    
    # For alerts, append distance for TTS
    if current_tts_language == "bangla":
        return ", ".join([f"{desc} {translate_text('VERY CLOSE', 'bn', for_tts=True)}" for desc in section_descriptions])
    else:
        return ", ".join(section_descriptions) + " is too close!" if len(section_descriptions) == 1 else ", ".join(section_descriptions) + " are too close!"

def describe_frame(objects_info):
    """Generate detailed description of current frame with natural phrasing"""
    formatted_objects, section_descriptions = format_object_list(objects_info, for_alert=False)
    
    if not section_descriptions:
        return translate_text("No objects detected.", "bn" if current_tts_language == "bangla" else "en", for_tts=True)
    
    return ", ".join(section_descriptions) + "."

# ----------------------------
# 8. Main Processing Loop
# ----------------------------
print("Dual-Mode System Active!")
print("Mode 1: Continuous Alerts (default) - Warns about close objects")
print("Mode 2: Frame Description - Describes everything in view")
print("Press 't' to toggle modes")
print("Press 'n' in Mode 2 to describe current frame")
print("Press 'b' to switch TTS language (English/Bangla)")
print("Press 'q' to quit")

frame_counter = 0
process_every_n_frames = 2
last_description = ""
last_alert = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_counter += 1
    if frame_counter % process_every_n_frames != 0:
        continue

    # Run models
    results = yolo_model(frame, imgsz=640, verbose=False, conf=0.6)
    depth_map = run_midas(frame)

    # Clean up old objects
    tracker.cleanup_old_objects()

    close_objects_info = []  # For Mode 1 alerts: (name, distance, section)
    all_objects_info = []    # For Mode 2 description: (name, distance, section)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            
            if x2 <= x1 or y2 <= y1 or score < 0.6:
                continue
            
            expanded_x1 = max(0, x1 - 5)
            expanded_y1 = max(0, y1 - 5)
            expanded_x2 = min(frame.shape[1], x2 + 5)
            expanded_y2 = min(frame.shape[0], y2 + 5)
            
            depth_roi = depth_map[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            if depth_roi.size == 0:
                continue
                
            current_depth = np.median(depth_roi)
            obj_id = tracker.get_object_id((x1, y1, x2, y2), int(class_id))
            smoothed_depth = tracker.update_depth(obj_id, current_depth)
            current_distance, color, is_close = depth_label(smoothed_depth)
            stable_distance = tracker.get_stable_label(obj_id, current_distance)
            
            object_name = yolo_model.names[int(class_id)]
            
            # Determine section (left/middle/right)
            x_center = (x1 + x2) // 2
            section = get_section(x_center, frame.shape[1])
            
            # Use English for display
            display_object_name = object_name
            display_distance = stable_distance
            display_section = "in front" if section == "middle" else f"on the {section}"
            
            # Add section to label (English for display)
            label = f"{display_object_name}: {display_distance} ({display_section})"
            
            # Store for both modes with section info (use original names for processing)
            all_objects_info.append((object_name, stable_distance, section))
            if is_close:  # Only close objects for alerts
                close_objects_info.append((object_name, stable_distance, section))
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, 
                         (x1, y1 - label_height - 5),
                         (x1 + label_width, y1),
                         color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # ----------------------------
    # 9. Mode-Specific Processing
    # ----------------------------
    current_time = time.time()
    
    if current_mode == SystemMode.CONTINUOUS_ALERT:
        # Mode 1: Continuous Alerts (only for close objects)
        alert_message = generate_alert_message(close_objects_info)
        if alert_message and current_time - last_alert_time > alert_cooldown and not tts_busy:
            print(f"ALERT: {alert_message if TERMINAL_IN_ENGLISH else translate_text(alert_message, 'bn', for_tts=True)}")
            last_alert = alert_message
            last_alert_time = current_time
            # Speak the alert
            speak_text(alert_message, current_tts_language)
            
        mode_text = "MODE 1: Continuous Alerts"
        close_count = len(close_objects_info)
        status_text = f"Close objects: {close_count}"
        
    else:
        # Mode 2: Frame Description (all objects)
        mode_text = "MODE 2: Frame Description"
        status_text = "Press 'n' to describe frame"
        
        # Show last description on screen (in English)
        if last_description:
            words = last_description.split()
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
                cv2.putText(frame, line, (10, 40 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ----------------------------
    # 10. Display and Controls
    # ----------------------------
    cv2.putText(frame, mode_text, (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, status_text, (10, frame.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Show current TTS language (in English)
    lang_text = f"TTS Language: {current_tts_language.upper()}"
    cv2.putText(frame, lang_text, (frame.shape[1] - 250, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    controls_text = "Controls: 't'=Toggle Mode | 'n'=Describe Frame | 'b'=Toggle Language | 'q'=Quit"
    cv2.putText(frame, controls_text, (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Advanced Object Detection System", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        current_mode = SystemMode.FRAME_DESCRIPTION if current_mode == SystemMode.CONTINUOUS_ALERT else SystemMode.CONTINUOUS_ALERT
        mode_name = "Frame Description" if current_mode == SystemMode.FRAME_DESCRIPTION else "Continuous Alerts"
        print(f"Switched to {mode_name} mode" if TERMINAL_IN_ENGLISH else translate_text(f"Switched to {mode_name} mode", "bn", for_tts=True))
        last_description = ""
    elif key == ord('n') and current_mode == SystemMode.FRAME_DESCRIPTION:
        last_description = describe_frame(all_objects_info)
        print(f"FRAME: {last_description if TERMINAL_IN_ENGLISH else translate_text(last_description, 'bn', for_tts=True)}")
        # Speak the description
        speak_text(last_description, current_tts_language)
    elif key == ord('b'):
        # Toggle TTS language
        current_tts_language = "bangla" if current_tts_language == "english" else "english"
        lang_message = f"Language switched to {current_tts_language}"
        print(translate_text(lang_message, "bn" if current_tts_language == "bangla" else "en", for_tts=True))
        # Speak confirmation
        speak_text(lang_message, current_tts_language)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("System shutdown complete." if TERMINAL_IN_ENGLISH else translate_text("System shutdown complete.", "bn", for_tts=True))