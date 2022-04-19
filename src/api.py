from uuid import uuid4
from threading import Thread

import json
import os
import errno
import cv2

from helpers import Timer
from generator import generate_all_videos, Image

from flask import Flask, send_from_directory, request, Response
app = Flask(__name__)


@app.route("/images", methods=["GET"])
def getImages():
    images = os.listdir("images")

    if "processing" in images:
        images.remove("processing")
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    return json.dumps({"images": images, "processing": os.listdir("images/processing")})


@app.route("/", defaults={"path": "src/index_additive.html"}, methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def getStatic(path):
    return send_from_directory("../", path)


@app.route("/exists/<path:image>", methods=["GET"])
def getExists(image):
    return json.dumps(os.path.isfile("images/" + image))


@app.route("/addImage", methods=["POST"])
def add_image():
    if "file" not in request.files:
        print("No file part")
        return Response(json.dumps({"error": "no file uploaded"}), status=400)
    file = request.files["file"]

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
