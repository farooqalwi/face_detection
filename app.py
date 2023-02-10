import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/api/face_detection", methods=["POST"])
def face_detection():
    try:
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Get the image data from the request
        img_data = request.data

        # Convert the image data to a NumPy array
        arr = np.frombuffer(img_data, np.uint8)

        # Decode the image from the NumPy array
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)

        # Return 1 if face is present, 0 otherwise
        if len(faces) == 0:
            return jsonify({"Result": "No face found in the image", "Code": 0})
        else:
            return jsonify({"Result": "Face found in the image", "Code": 1})
    except Exception:
        return jsonify({"Error": "No image was provided or Image has broken.", "Code": 400}), 400


if __name__ == "__main__":
    app.run()
