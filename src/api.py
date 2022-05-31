from time import sleep
from uuid import uuid4
from threading import Thread

import json
import os
import errno
import cv2
from cv2 import imshow
import numpy as np
from detector import detect

from helpers import Timer, raise_error_response
from generator import generate_all_videos, Image
from storage.image import list_images

from flask import Flask, send_from_directory, request, make_response, jsonify
app = Flask(__name__)


@app.route("/", defaults={"path": "src/index_additive.html"}, methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def getStatic(path):
    return send_from_directory("../", path)


@app.route("/exists/<path:image>", methods=["GET"])
def getExists(image):
    return json.dumps(os.path.isfile("images/" + image))


@app.route("/images", methods=["GET"])
def getImages():
    return list_images()


@app.route("/images", methods=["POST"])
def add_image():
    file = get_request_file("file")

    if file:
        filename = str(uuid4()) + "." + file.filename.split(".")[-1]
        filepath = "images/processing/" + filename
        file.save(filepath)

        to_img = Image(cv2.imread(filepath))
        if not to_img.contains_face:
            os.remove(filepath)
            return {"error": "no face detected"}
        Thread(target=handle_image_upload, args=[to_img, filename]).start()
        return {"response": "processing", "img_name": filename}

    return json.dumps({"error": "file is empty"})


def handle_image_upload(to_img, img_name):
    generate_all_videos(to_img, img_name)
    os.rename(f"images/processing/{img_name}", f"images/{img_name}")


@app.route("/images/<path:image>", methods=["GET"])
def get_image(image):
    return send_from_directory("../images", image)


@app.route("/images/detect", methods=["POST"])
def detect_image():
    img = get_request_image("file")
    r = detect(img)
    # return make_image_response(r)
    return jsonify(r)

def make_image_response(img):
    _, buffer = cv2.imencode('.png', img)
    response = make_response(buffer.tobytes())
    response.headers.set('Content-Type', 'image/png')
    return response


def get_request_image(file_key):
    file = get_request_file(file_key)
    nparr = np.fromfile(file, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)


def get_request_file(file_key):
    if file_key not in request.files:
        print("No file part")
        raise_error_response(
            f"expeced file under key [{file_key}]", status=400)
    return request.files[file_key]


@app.route("/times", methods=["GET"])
def get_times():
    return Timer.stat()


if __name__ == "__main__":
    # Create required dirs if they do not exist
    try:
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("images"):
            os.makedirs("images")
        if not os.path.exists("images/processing"):
            os.makedirs("images/processing")
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

    app.run(host="0.0.0.0", port=5000, debug=True)
