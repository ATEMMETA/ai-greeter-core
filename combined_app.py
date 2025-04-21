# combined_app.py
import cv2
import face_recognition
import numpy as np
import os
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://facegreeter.vercel.app"}})
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
images_path = "/content/images"
os.makedirs(images_path, exist_ok=True)
known_face_encodings = {}
for fname in os.listdir(images_path):
    if fname.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(fname)[0]
        image_path = os.path.join(images_path, fname)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings[name] = encodings[0]

connected_cameras = {}

@app.route('/connect_camera', methods=['POST'])
def connect_camera():
    try:
        data = request.json
        ip = data.get('ip')
        username = data.get('username', 'admin')
        password = data.get('password', '12345')
        wifi_ssid = data.get('wifi_ssid', '')
        wifi_password = data.get('wifi_password', '')
        logger.info(f"Attempting to connect to camera at {ip}")

        import re
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            return jsonify({"success": False, "error": "Invalid IP address"}), 400

        rtsp_url = f"rtsp://{username}:{password}@{ip}:554/live/ch00_0"
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            logger.warning(f"Failed to connect to camera at {ip}")
            return jsonify({"success": False, "error": "Failed to connect to camera"}), 400
        cap.release()

        if wifi_ssid and wifi_password:
            try:
                wifi_response = requests.post(
                    f"http://{ip}/set_wifi",
                    data={"ssid": wifi_ssid, "password": wifi_password},
                    timeout=5
                )
                if wifi_response.status_code != 200:
                    logger.warning(f"Failed to configure Wi-Fi for {ip}")
            except requests.RequestException as e:
                logger.error(f"Wi-Fi configuration error: {str(e)}")

        connected_cameras[ip] = rtsp_url
        logger.info(f"Successfully connected to camera at {ip}")

        vercel_url = os.getenv("VERCEL_API_BASE_URL", "https://facegreeter.vercel.app")
        requests.post(
            f"{vercel_url}/api/camera_connected",
            json={"ip": ip, "rtsp_url": rtsp_url, "model": "V380"}
        )

        return jsonify({"success": True, "rtsp_url": rtsp_url, "error": None})
    except Exception as e:
        logger.error(f"Error connecting to camera: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/video_feed', methods=['GET'])
def video_feed():
    try:
        rtsp_url = request.args.get('rtsp')
        if not rtsp_url or rtsp_url not in connected_cameras.values():
            return jsonify({"success": False, "error": "Invalid or unconnected camera"}), 400

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open camera stream"}), 500

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
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(frame_rgb)
                        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                            name = "Unknown"
                            matches = face_recognition.compare_faces(list(known_face_encodings.values()), encoding)
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = list(known_face_encodings.keys())[first_match_index]
                            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
                        ret, buffer = cv2.imencode(".jpg", frame)
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                prev_frame = gray
            cap.release()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error streaming video: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.json
        rtsp_url = data.get('rtsp_url')
        if not rtsp_url or rtsp_url not in connected_cameras.values():
            return jsonify({"success": False, "error": "Invalid or unconnected RTSP URL"}), 400

        logger.info(f"Detecting faces for RTSP: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open camera stream"}), 500

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"success": False, "error": "Failed to capture frame"}), 500

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        face_names = []
        for encoding in face_encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(list(known_face_encodings.values()), encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = list(known_face_encodings.keys())[first_match_index]
            face_names.append(name)
        logger.info(f"Detected {len(face_names)} faces")
        return jsonify({"success": True, "locations": face_locations, "names": face_names})
    except Exception as e:
        logger.error(f"Error detecting face: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/customer_insights', methods=['POST'])
def customer_insights():
    try:
        data = request.json
        face_names = data.get('face_names', [])
        logger.info(f"Processing customer insights for faces: {face_names}")

        insights = []
        for name in face_names:
            prompt = f"Greet {name} warmly as a returning customer." if name != "Unknown" else "Greet an unknown visitor politely and offer assistance."
            try:
                response = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROK_API_KEY}"},
                    json={"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100},
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
                insights.append({
                    "name": name,
                    "status": "Returning customer" if name != "Unknown" else "New visitor",
                    "greeting": reply
                })
            except Exception as e:
                logger.error(f"Grok API error for {name}: {str(e)}")
                insights.append({
                    "name": name,
                    "status": "Error",
                    "greeting": f"Welcome{' back, ' + name if name != 'Unknown' else '!'}"
                })

        return jsonify({"success": True, "insights": insights})
    except Exception as e:
        logger.error(f"Error generating customer insights: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        data = request.json
        name = data.get('name')
        image_data = data.get('image_data')
        if not name or not image_data:
            return jsonify({"success": False, "error": "Missing name or image data"}), 400

        image_array = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(frame_rgb)
        if not encodings:
            return jsonify({"success": False, "error": "No face detected"}), 400

        image_path = os.path.join(images_path, f"{name}.jpg")
        cv2.imwrite(image_path, image_array)
        known_face_encodings[name] = encodings[0]
        logger.info(f"Added face for {name}")
        return jsonify({"success": True, "error": None})
    except Exception as e:
        logger.error(f"Error adding face: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
