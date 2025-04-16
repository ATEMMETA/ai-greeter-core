import sys
assert sys.version_info[:2] == (3, 10), "Use Python 3.10"
import os
from pyngrok import ngrok
from flask import Flask, request, Response, jsonify
from process_frame import process_frame, add_face
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

@app.route("/video_feed")
def video_feed():
    def generate():
        for frame_bytes in process_frame():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/add_face", methods=["POST"])
def add_face_route():
    try:
        name = request.form["name"]
        image = request.files["image"]
        success, error = add_face(name, image.read())
        return jsonify({"success": success, "error": error}), 200 if success else 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    public_url = ngrok.connect(5000).public_url
    print(f"API at: {public_url}")
    app.run(host="0.0.0.0", port=5000)
