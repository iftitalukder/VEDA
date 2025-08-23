import cv2
import numpy as np
import tensorflow as tf

# Load COCO labels
with open("models/coco_labels.txt", "r") as f:
    coco_labels = [line.strip() for line in f.readlines()]

# Load YOLOv11x TFLite model
model_path = "models/yolo11x_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height, input_width = input_details[0]['shape'][1], input_details[0]['shape'][2]

print(f"YOLOv11x model loaded. Input size: {input_width}x{input_height}")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, _ = frame.shape

    # Prepare input
    input_frame = cv2.resize(frame, (input_width, input_height))
    input_frame = np.expand_dims(input_frame.astype(np.float32), axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # [N,85]

    for det in output:
        if len(det) < 6:
            continue

        x, y, w, h = det[:4]
        objectness = det[4]
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        score = objectness * class_scores[class_id]

        if score < 0.5:
            continue

        # Convert to frame coordinates
        xmin = int((x - w/2) * width)
        xmax = int((x + w/2) * width)
        ymin = int((y - h/2) * height)
        ymax = int((y + h/2) * height)

        label = coco_labels[class_id] if class_id < len(coco_labels) else str(class_id)

        # Draw box and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv11x Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test finished.")
