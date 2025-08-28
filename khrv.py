
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
from gtts import gTTS
import pygame
import io
import threading
import re
import psutil
import gc
from collections import OrderedDict



# Load Models
yolo_model = YOLO("models/yolo11s")
print("YOLO11s loaded!")

midas = tf.lite.Interpreter(model_path="models/midas.tflite", num_threads=2)
midas.allocate_tensors()
input_details = midas.get_input_details()
output_details = midas.get_output_details()
midas_h, midas_w = 256, 256
print(f"MiDaS TFLite loaded. Input size: {midas_w}x{midas_h}")

# Warm-up models
def warm_up_models():
    dummy_frame = np.zeros((128, 96, 3), dtype=np.uint8)
    yolo_model(dummy_frame, imgsz=128, verbose=False, conf=0.3)
    frame_resized = cv2.resize(dummy_frame, (midas_w, midas_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_data = np.expand_dims(frame_rgb, axis=0)
    midas.set_tensor(input_details[0]['index'], input_data)
    midas.invoke()
    print("Models warmed up!")
warm_up_models()

# TTS Setup
pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=4096)  # Increased buffer size
current_tts_language = "english"
tts_enabled = True
last_spoken_text = ""
tts_lock = threading.Lock()
tts_busy = False
last_alert_time = 0
last_speech_time = 0
alert_cooldown = 3.0
speech_interval = 3.0  # Increased to 3.0s for both modes

# Limit TTS cache to 100 entries
tts_cache = OrderedDict()
MAX_CACHE_SIZE = 100
def cache_tts(text, lang):
    if (text, lang) not in tts_cache:
        try:
            tts = gTTS(text=translate_to_bangla(text) if lang == "bangla" else text, 
                       lang="bn" if lang == "bangla" else "en", slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            tts_cache[(text, lang)] = mp3_fp
            if len(tts_cache) > MAX_CACHE_SIZE:
                tts_cache.popitem(last=False)
            print(f"TTS cached: {text} ({lang})")
        except Exception as e:
            print(f"TTS Cache Error for {text} ({lang}): {e}")
            return None
    return tts_cache.get((text, lang))

for lang in ["english", "bangla"]:
    cache_tts("No objects detected", lang)
    cache_tts("Switched to Alert Mode", lang)
    cache_tts("Switched to Description Mode", lang)
    cache_tts("Language switched to english", lang)
    cache_tts("Language switched to bangla", lang)

def speak_text(text, language="english"):
    global tts_busy, last_spoken_text, last_speech_time
    if not text or text == last_spoken_text:
        return
    current_time = time.time()
    if current_time - last_speech_time < speech_interval:
        print(f"TTS skipped: {text} (too soon)")
        return
    try:
        with tts_lock:
            if tts_busy:
                print(f"TTS busy: {text} skipped")
                return
            tts_busy = True
            last_spoken_text = text
            last_speech_time = current_time
            def tts_thread():
                global tts_busy
                try:
                    if not pygame.mixer.get_init():
                        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=4096)
                    pygame.mixer.music.stop()  # Ensure mixer is clear
                    pygame.mixer.music.unload()  # Unload previous audio
                    cached_audio = cache_tts(text, language)
                    if cached_audio:
                        cached_audio.seek(0)
                        pygame.mixer.music.load(cached_audio)
                    else:
                        text_to_speak = translate_to_bangla(text) if language == "bangla" else text
                        tts = gTTS(text=text_to_speak, lang="bn" if lang == "bangla" else "en", slow=False)
                        mp3_fp = io.BytesIO()
                        tts.write_to_fp(mp3_fp)
                        mp3_fp.seek(0)
                        pygame.mixer.music.load(mp3_fp)
                    pygame.mixer.music.play()
                    print(f"TTS playing: {text} ({language})")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)  # Reduced sleep for smoother checking
                except Exception as e:
                    print(f"TTS Error: {e}")
                finally:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    tts_busy = False
            thread = threading.Thread(target=tts_thread, daemon=False)
            thread.start()
    except Exception as e:
        print(f"TTS Outer Error: {e}")
        tts_busy = False

def translate_to_bangla(text):
    TRANSLATIONS = {
        "Warning": "সতর্কতা",
        "is too close": "খুব কাছে",
        "are too close": "খুব কাছে",
        "No objects detected": "কোনো বস্তু সনাক্ত করা হয়নি",
        "Switched to Alert Mode": "সতর্ক মোডে স্যুইচ করা হয়েছে",
        "Switched to Description Mode": "বর্ণনা মোডে স্যুইচ করা হয়েছে",
        "Language switched to bangla": "ভাষা বাংলায় স্যুইচ করা হয়েছে",
        "Language switched to english": "ভাষা ইংরেজিতে স্যুইচ করা হয়েছে",
        "VERY CLOSE": "খুব কাছে",
        "CLOSE": "কাছে",
        "MEDIUM": "মাঝারি",
        "FAR": "দূরে",
        "straight ahead": "সোজা সামনে",
        "degree left": "ডিগ্রি বামে",
        "degree right": "ডিগ্রি ডানে",
    }
    result = text
    for eng, ban in TRANSLATIONS.items():
        result = result.replace(eng, ban)
    result = re.sub(r'(\d+) degree left', r'\1 ডিগ্রি বামে', result)
    result = re.sub(r'(\d+) degree right', r'\1 ডিগ্রি ডানে', result)
    result = re.sub(r'(\d+) (\w+)s', r'\1 \2s', result)
    result = result.replace("a ", "")
    return result

# Video Capture Setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open USB camera!")
    pygame.mixer.quit()
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 96)
cap.set(cv2.CAP_PROP_FPS, 5)
print("USB camera initialized at 128x96, 5 FPS")

# Mode Management
class SystemMode:
    ALERT_MODE = 1
    DESCRIPTION_MODE = 2

current_mode = SystemMode.ALERT_MODE

# Angular Positioning
MAX_ANGLE = 45
STRAIGHT_THRESHOLD = 5

def calculate_angle(x_center, frame_width):
    center_x = frame_width / 2
    offset = x_center - center_x
    normalized_offset = offset / center_x
    angle = normalized_offset * MAX_ANGLE
    return angle

def get_angular_position(angle):
    if abs(angle) <= STRAIGHT_THRESHOLD:
        return "straight ahead"
    elif angle < 0:
        return f"{abs(angle):.0f} degree left"
    else:
        return f"{angle:.0f} degree right"

# Object Tracker
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
                abs(center_x - prev_center_x) < 30 and 
                abs(center_y - prev_center_y) < 30):
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

# Helper Functions
def run_midas(frame, run=True):
    if not run:
        return None
    frame_resized = cv2.resize(frame, (midas_w, midas_h))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_data = np.expand_dims(frame_rgb, axis=0)
    midas.set_tensor(input_details[0]['index'], input_data)
    midas.invoke()
    depth_map = midas.get_tensor(output_details[0]['index'])[0, :, :, 0]
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    return depth_normalized

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
    object_groups = {}
    for obj_name, distance, angle in objects_info:
        if for_alert and distance not in ["VERY CLOSE", "CLOSE", "MEDIUM"]:
            continue
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
            object_list.append(f"a {obj_name}" if count == 1 and for_alert else f"{count} {obj_name}s" if count > 1 else f"{obj_name}")
        objects_str = " and ".join(object_list) if len(object_list) <= 2 else ", ".join(object_list[:-1]) + f", and {object_list[-1]}"
        position_descriptions.append(f"{objects_str} {position}")
    return position_descriptions

def generate_alert_message(close_objects_info):
    position_descriptions = format_object_list(close_objects_info, for_alert=True)
    if not position_descriptions:
        return None
    return f"Warning: {position_descriptions[0] if len(position_descriptions) == 1 else ', '.join(position_descriptions)} is too close!"

def describe_frame(objects_info):
    position_descriptions = format_object_list(objects_info, for_alert=False)
    if not position_descriptions:
        return "No objects detected."
    return ", ".join(position_descriptions[:-1]) + f", and {position_descriptions[-1]}" if len(position_descriptions) > 1 else position_descriptions[0]

# Main Loop
skip_frames = 4
midas_skip = 16
frame_count = 0
current_skip = 0
previous_all_objects_info = []
previous_close_objects_info = []
previous_depth_map = None
last_description_message = ""  # Track last description to avoid repeats

def run_yolo(frame, results_out):
    results_out[0] = yolo_model(frame, imgsz=128, verbose=False, conf=0.3)

def run_midas_thread(frame, depth_out, run=True):
    depth_out[0] = run_midas(frame, run)

print("Video Testing Mode: 128x96, YOLO11n NCNN (128x128), MiDaS Small INT8 (256x256)")
print("Angular Range: ±45° (straight ahead: ±5°)")
print("Headless, runs until Ctrl+C")
print("Note: cv2.waitKey may not work in truly headless mode; consider alternative input methods if no display is available")

try:
    while True:
        frame_start_time = time.time()
        ret, original_frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera!")
            break
        original_height, original_width = original_frame.shape[:2]
        
        yolo_results = [None]
        midas_depth = [None]

        if current_skip % skip_frames == 0:
            yolo_thread = threading.Thread(target=run_yolo, args=(original_frame, yolo_results))
            midas_run = current_skip % midas_skip == 0
            midas_thread = threading.Thread(target=run_midas_thread, args=(original_frame, midas_depth, midas_run))
            yolo_thread.start()
            midas_thread.start()
            yolo_thread.join()
            midas_thread.join()
            results = yolo_results[0]
            depth_map = midas_depth[0] if midas_run else previous_depth_map

            all_objects_info = []
            close_objects_info = []
            if results is not None:
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        if x2 <= x1 or y2 <= y1 or score < 0.3:
                            continue
                        if depth_map is not None:
                            depth_roi = depth_map[y1:y2, x1:x2]
                            if depth_roi.size == 0:
                                print(f"Warning: Empty depth ROI for box {box}")
                                continue
                            current_depth = np.median(depth_roi)
                            current_distance, is_close = depth_label(current_depth)
                            object_name = yolo_model.names[int(class_id)]
                            x_center = (x1 + x2) // 2
                            angle = calculate_angle(x_center, original_width)
                            all_objects_info.append((object_name, current_distance, angle))
                            if is_close:
                                close_objects_info.append((object_name, current_distance, angle))
            
            previous_all_objects_info = all_objects_info
            previous_close_objects_info = close_objects_info
            previous_depth_map = depth_map
            tracker.cleanup_old_objects()
        
        current_skip += 1
        current_time = time.time()

        if current_mode == SystemMode.ALERT_MODE:
            alert_message = generate_alert_message(previous_close_objects_info)
            if alert_message and current_time - last_alert_time > alert_cooldown and not tts_busy and tts_enabled:
                print(f"ALERT: {alert_message}")
                speak_text(alert_message, current_tts_language)
                last_alert_time = current_time
        else:
            description_message = describe_frame(previous_all_objects_info)
            if description_message != last_description_message:  # Only speak if message changed
                print(f"DESCRIPTION: {description_message}")
                if tts_enabled:
                    speak_text(description_message, current_tts_language)
                last_description_message = description_message

        # Status update
        if frame_count % 10 == 0:
            mode_text = "ALERT MODE" if current_mode == SystemMode.ALERT_MODE else "DESCRIPTION MODE"
            lang_text = f"TTS: {current_tts_language.upper()} {'ON' if tts_enabled else 'OFF'}"
            fps = 1000 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
            print(f"Frame {frame_count}: {mode_text}, {lang_text}, FPS: {fps:.2f}, CPU Usage: {psutil.cpu_percent()}%")

        # Keyboard input for mode/language switch
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            current_mode = SystemMode.DESCRIPTION_MODE if current_mode == SystemMode.ALERT_MODE else SystemMode.ALERT_MODE
            mode_name = "Description Mode" if current_mode == SystemMode.DESCRIPTION_MODE else "Alert Mode"
            print(f"Switched to {mode_name}")
            if tts_enabled:
                speak_text(f"Switched to {mode_name}", current_tts_language)
        elif key == ord('l'):
            current_tts_language = "bangla" if current_tts_language == "english" else "english"
            print(f"TTS language switched to {current_tts_language}")
            if tts_enabled:
                speak_text(f"Language switched to {current_tts_language}", current_tts_language)

        gc.collect()
        elapsed = (time.time() - frame_start_time)
        if elapsed < 0.2 and elapsed >= 0:
            time.sleep(0.2 - elapsed)
        frame_count += 1

except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C). Releasing camera...")
finally:
    cap.release()
    pygame.mixer.quit()
    print("Video testing completed! Camera released.")
