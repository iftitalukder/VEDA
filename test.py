import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# 1. Initialize the TFLite Model
# ----------------------------
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="midas.tflite")  # Make sure the path is correct
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

print(f"Model expects input shape: {input_shape}")
print("Model loaded successfully!")

# ----------------------------
# 2. Initialize USB Camera
# ----------------------------
cap = cv2.VideoCapture(2)  # 0 is usually the default USB camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit")

# ----------------------------
# 3. Main Processing Loop
# ----------------------------
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Preprocess the frame for the model
    # a) Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # b) Resize to model's expected input size
    model_input = cv2.resize(frame_rgb, (input_width, input_height))
    # c) Normalize pixel values to [0, 1] and add batch dimension
    model_input = np.expand_dims(model_input.astype(np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], model_input)
    interpreter.invoke()
    
    # Get the depth map output
    depth_pred = interpreter.get_tensor(output_details[0]['index'])
    
    # Remove batch dimension and squeeze
    depth_map = np.squeeze(depth_pred)
    
    # Normalize the depth map for visualization (0-255)
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 0:
        depth_map_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    else:
        depth_map_vis = np.zeros_like(depth_map, dtype=np.uint8)
    
    # Apply a color map (jet is good for depth)
    depth_colormap = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_JET)
    
    # Resize depth colormap to match original camera frame size
    depth_colormap = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))
    
    # ----------------------------
    # 4. Display Results
    # ----------------------------
    # Option 1: Side-by-side view
    combined = np.hstack((frame, depth_colormap))
    cv2.imshow('Live Camera | Depth Estimation', combined)
    
    # Option 2: Overlay (transparent)
    # overlay = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)
    # cv2.imshow('Depth Overlay', overlay)

    # Check for quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# 5. Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
print("Test completed.")