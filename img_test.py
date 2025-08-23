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
import glob
import math

# ----------------------------
# 1. Display Configuration
# ----------------------------
MAX_DISPLAY_WIDTH = 1280  # Maximum width for display
MAX_DISPLAY_HEIGHT = 720  # Maximum height for display
LABEL_SCALE = 0.4         # Scale factor for labels
BOX_THICKNESS = 1      # Thickness of bounding boxes

# ----------------------------
# 2. Load Models
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
# 3. TTS Setup
# ----------------------------
pygame.mixer.init()
current_tts_language = "english"
tts_enabled = True
last_spoken_text = ""

def speak_text(text, language="english"):
    """Convert text to speech in background thread - speaks only once per unique text"""
    global last_spoken_text
    
    if text == last_spoken_text or not tts_enabled:
        return
        
    last_spoken_text = text
    
    def tts_thread():
        try:
            lang_code = "bn" if language == "bangla" else "en"
            tts = gTTS(text=text, lang=lang_code, slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            pygame.mixer.music.load(mp3_fp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except Exception as e:
            print(f"TTS Error: {e}")
    
    thread = threading.Thread(target=tts_thread)
    thread.daemon = True
    thread.start()

# ----------------------------
# 4. Image Testing Setup
# ----------------------------
test_images_folder = "test_images"
image_files = sorted(glob.glob(os.path.join(test_images_folder, "*.jpg")) + 
                   glob.glob(os.path.join(test_images_folder, "*.png")) +
                   glob.glob(os.path.join(test_images_folder, "*.jpeg")))

if not image_files:
    print(f"No images found in {test_images_folder} folder!")
    print("Please add images and restart.")
    exit()

current_image_index = 0
print(f"Found {len(image_files)} test images")

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
MAX_ANGLE = 45  # Maximum 100° left/right
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
# 7. Image Resizing Functions
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
# 8. Object Tracker
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
        # Reset for new image
        self.objects = {}
        self.next_id = 0

tracker = ObjectTracker()

# ----------------------------
# 9. Helper Functions
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
            # For alerts, only include if distance indicates closeness
            if distance in ["VERY CLOSE", "CLOSE", "MEDIUM"]:
                position = get_angular_position(angle)
                if position not in object_groups:
                    object_groups[position] = {}
                if obj_name not in object_groups[position]:
                    object_groups[position][obj_name] = 0
                object_groups[position][obj_name] += 1
        else:
            # For description, include all objects
            position = get_angular_position(angle)
            if position not in object_groups:
                object_groups[position] = {}
            if obj_name not in object_groups[position]:
                object_groups[position][obj_name] = 0
            object_groups[position][obj_name] += 1
    
    if not object_groups:
        return None
    
    # Build formatted descriptions for each position
    position_descriptions = []
    
    for position, objects in object_groups.items():
        object_list = []
        for obj_name, count in objects.items():
            if count == 1:
                object_list.append(f"a {obj_name}" if for_alert else f"1 {obj_name}")
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
    """Generate detailed description of all objects with angular position"""
    position_descriptions = format_object_list(objects_info, for_alert=False)
    
    if not position_descriptions:
        return "No objects detected."
    
    if len(position_descriptions) == 1:
        return f"There is {position_descriptions[0]}."
    else:
        return f"There are {', '.join(position_descriptions)}."

def load_next_image():
    """Load the next image in sequence"""
    global current_image_index, last_spoken_text
    current_image_index = (current_image_index + 1) % len(image_files)
    last_spoken_text = ""  # Reset TTS memory
    tracker.cleanup_old_objects()
    return load_current_image()

def load_previous_image():
    """Load the previous image in sequence"""
    global current_image_index, last_spoken_text
    current_image_index = (current_image_index - 1) % len(image_files)
    last_spoken_text = ""  # Reset TTS memory
    tracker.cleanup_old_objects()
    return load_current_image()

def load_current_image():
    """Load the current image"""
    image_path = image_files[current_image_index]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Resize for display
    display_image = resize_image_to_fit(image)
    return image, display_image

def save_result_image(original_image, output_folder="results"):
    """Save the processed image with detections"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_name = os.path.basename(image_files[current_image_index])
    output_path = os.path.join(output_folder, f"result_{image_name}")
    
    # Use the original image for saving (high quality)
    result_image = original_image.copy()
    height, width = result_image.shape[:2]
    
    # Process on original image for high-quality saving
    results = yolo_model(result_image, imgsz=640, verbose=False, conf=0.6)
    depth_map = run_midas(result_image)
    
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
            
            # Larger labels for saved images
            label = f"{object_name}: {current_distance} ({angular_pos})"
            
            # Draw on original image
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, BOX_THICKNESS * 2)
            
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.5, BOX_THICKNESS * 2
            )
            cv2.rectangle(result_image, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.5, (0, 0, 0), BOX_THICKNESS * 2)
    
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")

# ----------------------------
# 10. Main Processing Loop
# ----------------------------
print("Image Testing Mode with Angular Positioning Active!")
print(f"Loaded {len(image_files)} images from {test_images_folder}")
print("Display Resolution: 1280x720 (images will be scaled to fit)")
print("Angular Range: ±45° (straight ahead: ±5°)")
print("\nControls:")
print("'m' - Switch between Alert Mode and Description Mode")
print("'d' - Describe current frame")
print("'n' - Next image")
print("'p' - Previous image") 
print("'b' - Switch TTS language (English/Bangla)")
print("'s' - Toggle TTS on/off")
print("'w' - Save result image")
print("'q' - Quit")

original_image, current_display_image = load_current_image()
if original_image is None:
    exit()

while True:
    display_frame = current_display_image.copy()
    original_height, original_width = original_image.shape[:2]
    display_height, display_width = display_frame.shape[:2]
    
    # Calculate scale factor for coordinate conversion
    scale_x = display_width / original_width
    scale_y = display_height / original_height
    
    # Run models on original image for accuracy
    results = yolo_model(original_image, imgsz=640, verbose=False, conf=0.6)
    depth_map = run_midas(original_image)

    close_objects_info = []  # For Alert Mode: (name, distance, angle)
    all_objects_info = []    # For Description Mode: (name, distance, angle)

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
            
            # Draw on display image
            cv2.rectangle(display_frame, (display_x1, display_y1), (display_x2, display_y2), color, BOX_THICKNESS)
            
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, BOX_THICKNESS
            )
            cv2.rectangle(display_frame, (display_x1, display_y1 - label_height - 10), 
                         (display_x1 + label_width, display_y1), color, -1)
            cv2.putText(display_frame, label, (display_x1, display_y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 0, 0), BOX_THICKNESS)

    # Generate messages based on current mode
    if current_mode == SystemMode.ALERT_MODE:
        alert_message = generate_alert_message(close_objects_info)
        description_message = alert_message
    else:
        alert_message = None
        description_message = describe_frame(all_objects_info)
    
    # Display info
    image_name = os.path.basename(image_files[current_image_index])
    cv2.putText(display_frame, f"Image: {image_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    mode_text = "ALERT MODE" if current_mode == SystemMode.ALERT_MODE else "DESCRIPTION MODE"
    cv2.putText(display_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    lang_text = f"TTS: {current_tts_language.upper()} {'ON' if tts_enabled else 'OFF'}"
    cv2.putText(display_frame, lang_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
    
    controls = "Controls: m=Mode d=Describe n=Next p=Prev b=Lang s=TTS w=Save q=Quit"
    cv2.putText(display_frame, controls, (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow("Angular Object Detection", display_frame)
    
    key = cv2.waitKey(0) & 0xFF  # Wait for key press
    
    if key == ord('q'):
        break
    elif key == ord('n'):
        original_image, current_display_image = load_next_image()
    elif key == ord('p'):
        original_image, current_display_image = load_previous_image()
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
        save_result_image(original_image)
        print("Image saved with high-quality labels!")

cv2.destroyAllWindows()
print("Image testing completed!")