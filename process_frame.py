import cv2
import mediapipe as mp
import numpy as np
import os
from dotenv import load_dotenv
import requests
from google.cloud import texttospeech
import time

load_dotenv()
V380_RTSP_URL = os.getenv("V380_RTSP_URL", "rtsp://admin:12345@192.168.1.100:554/live/ch00_0")
GROK_API_KEY = os.getenv("GROK_API_KEY")
FIREBASE_URL = os.getenv("FIREBASE_URL", "https://your-project.web.app")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
tts_client = texttospeech.TextToSpeechClient()

cap = cv2.VideoCapture(V380_RTSP_URL)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

images_path = "images"
if not os.path.exists(images_path):
    os.makedirs(images_path)
known_faces = {}
for fname in os.listdir(images_path):
    if fname.endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(fname)[0]
        known_faces[name] = os.path.join(images_path, fname)

def get_ai_response(name, is_known=True):
    prompt = f"Greet {name} warmly as a returning customer." if is_known else "Greet an unknown visitor politely and offer assistance."
    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}"},
            json={"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100},
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        synthesis_input = texttospeech.SynthesisInput(text=reply)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        tts_response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open("welcome.mp3", "wb") as out:
            out.write(tts_response.audio_content)
        return reply, True
    except Exception as e:
        print(f"AI/TTS error: {e}")
        return f"Welcome{' back, ' + name if is_known else '!'}", False

def detect_faces(frame):
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
    return face_locations, face_names

def process_frame():
    prev_frame = None
    last_name = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion = np.mean(diff) > 5  # Tune threshold
            if motion:
                face_locations, face_names = detect_faces(frame)
                for (y1, x2, y2, x1), name in zip(face_locations, face_names):
                    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    if name != last_name:
                        if name == "Unknown":
                            requests.post(f"{FIREBASE_URL}/log_unknown")
                        reply, success = get_ai_response(name, is_known=name != "Unknown")
                        if success:
                            requests.post(f"{FIREBASE_URL}/update_audio", files={'audio': open('welcome.mp3', 'rb')})
                        last_name = name
                if face_locations:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield buffer.tobytes()
        prev_frame = gray
        time.sleep(1)  # 1 FPS

def add_face(name, image_data):
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    frame_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    if results.detections:
        image_path = os.path.join(images_path, f"{name}.jpg")
        cv2.imwrite(image_path, image_array)
        known_faces[name] = image_path
        return True, None
    return False, "No face detected"

if __name__ == "__main__":
    for frame_bytes in process_frame():
        pass
