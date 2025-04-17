# Create combined_app.py
%%writefile /content/combined_app.py
import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Face Detection Setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
images_path = "/content/images"
os.makedirs(images_path, exist_ok=True)
known_faces = {}
for fname in os.listdir(images_path):
    if fname.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(fname)[0]
        known_faces[name] = os.path.join(images_path, fname)

@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        name = request.form['name']
        image = request.files['image']
        logger.info(f"Adding face: {name}")
        image_data = image.read()
        image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            image_path = os.path.join(images_path, f"{name}.jpg")
            cv2.imwrite(image_path, image_array)
            known_faces[name] = image_path
            logger.info(f"Face saved: {image_path}")
            return jsonify({"success": True, "error": None})
        logger.warning("No face detected")
        return jsonify({"success": False, "error": "No face detected"}), 400
    except Exception as e:
        logger.error(f"Error adding face: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        logger.info("Detecting faces")
        image = request.files['image']
        image_data = image.read()
        frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        face_locations = []
        face_names = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
                face_locations.append((y1, x2, y2, x1))
                face_img = frame[y1:y2, x1:x2]
                name = "Unknown"
                for known_name, known_path in known_faces.items():
                    known_img = cv2.imread(known_path)
                    if known_img is not None and face_img.shape == known_img.shape:
                        diff = np.mean((face_img - known_img) ** 2)
                        if diff < 1000:
                            name = known_name
                            break
                face_names.append(name)
            logger.info(f"Detected {len(face_names)} faces")
        return jsonify({"locations": face_locations, "names": face_names})
    except Exception as e:
        logger.error(f"Error detecting face: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Video Stream Setup
cap = cv2.VideoCapture(0)  # Use webcam fallback
if not cap.isOpened():
    logger.error("Cannot open camera")
    raise IOError("Cannot open camera")

def generate():
    prev_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            break
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion = np.mean(diff) > 5
            if motion:
                ret, buffer = cv2.imencode(".jpg", frame)
                files = {"image": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                try:
                    response = requests.post(f"{request.url_root}detect_face", files=files)
                    data = response.json()
                    for (y1, x2, y2, x1), name in zip(data["locations"], data["names"]):
                        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                except Exception as e:
                    logger.error(f"Face detection error: {str(e)}")
                ret, buffer = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        prev_frame = gray

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
