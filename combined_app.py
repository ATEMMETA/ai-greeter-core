from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://facegreeter.vercel.app"}})

@app.route('/hello', methods=['GET'])
def hello():
    logger.info("Received request to /hello")
    return jsonify({"success": True, "message": "Hello from Flask!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
