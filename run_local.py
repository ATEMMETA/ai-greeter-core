import os
import uvicorn
from pyngrok import ngrok
import threading
from face_detection import app as face_app
from video_stream import app as video_app

def run_face_detection():
    public_url = ngrok.connect(8000).public_url
    print(f"Face Detection API at: {public_url}")
    uvicorn.run(face_app, host="0.0.0.0", port=8000)

def run_video_stream():
    public_url = ngrok.connect(5000).public_url
    print(f"Video Stream API at: {public_url}")
    uvicorn.run(video_app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    os.environ["NGROK_AUTHTOKEN"] = "your_ngrok_token"
    os.environ["V380_RTSP_URL"] = "rtsp://admin:12345@your.ip:554/live/ch00_0"
    os.environ["VERCEL_URL"] = "https://facegreeter-git-main-lexas-projects-0c10021c.vercel.app"
    os.environ["FACE_DETECTION_URL"] = "http://localhost:8000"
    threading.Thread(target=run_face_detection, daemon=True).start()
    run_video_stream()
