from traceback import format_exc
from uuid import uuid4
from threading import Thread

import json
import os
import errno
import cv2
import numpy as np
from detector import detect, add_new_template as register_detector_template

from helpers import Timer, raise_error_response
from storage.video_generator import ensure_archive_video, ensure_video, generate_one_per_emotion, Image, set_main_version
from storage.image import list_images, read_cv2_image
from storage.landmark import dict_landmarks_meta

from flask import Flask, send_from_directory, request, make_response, jsonify

app = Flask(__name__)


@app.route("/", defaults={"path": "index_additive.html"}, methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def getStatic(path):
    return send_from_directory("static", path)


@app.route("/exists/<path:img_id>", methods=["GET"])
def getExists(img_id):
    for fname in os.listdir("images/"):
        if fname.startswith(img_id):
            return "true"
    return "false"


@app.route("/landmarks", methods=["GET"])
def getLandmarks():
    return dict_landmarks_meta()


@app.route("/images", methods=["GET"])
def getImages():
    return list_images()


@app.route("/images", methods=["POST"])
def add_image():
    file = get_request_file("file")

    if file:
        img_id = str(uuid4())
        filename = img_id + "." + file.filename.split(".")[-1]
        filepath = "images/processing/" + filename
        file.save(filepath)

        img = Image(img_id, cv2.imread(filepath))
        if not img.contains_face:
            os.remove(filepath)
            return {"error": "no face detected"}
        Thread(target=handle_image_upload, args=[img_id, img]).start()
        return {"response": "processing", "img_id": img_id}

    return json.dumps({"error": "file is empty"})


def handle_image_upload(img_id: str, img: Image):
    generate_one_per_emotion(img_id, img)
    for fname in os.listdir("images/processing/"):
        if fname.startswith(img_id):
            os.rename(f"images/processing/{fname}", f"images/{fname}")
            break
    register_detector_template(cv2.cvtColor(
        img.img_cpu, cv2.COLOR_BGR2GRAY), img_id)


@app.route("/images/<string:img_id>", methods=["GET"])
def get_image(img_id):
    img = read_cv2_image(img_id)
    return make_image_response(img)


@app.route("/images/processing/<string:img_id>", methods=["GET"])
def get_processing_image(img_id):
    img = read_cv2_image(img_id, path="images/processing")
    return make_image_response(img)


@app.route("/images/<string:img_id>/points", methods=["GET"])
def draw_face_points(img_id):
    img = read_cv2_image(img_id)
    return make_image_response(Image(img_id, img).draw())

@app.route("/output/<string:img_id>/<string:emotion>", methods=["GET"])
def get_output(img_id, emotion):
    '''
    get_output responds with set emotion of given img_id
    '''
    # omit file extension
    if '.' in img_id:
        img_id = '.'.join(img_id.split('.')[:-1])

    ensure_video(img_id, emotion)

    return send_from_directory("../output", f"{img_id}_{emotion}.mp4")


used_formIDs = set()
@app.route("/output/<string:img_id>/<string:emotion>/<string:version>", methods=["GET"])
def get_output_versioned(img_id, emotion, version):
    '''
    get_output_versioned responds with set emotion version and composition of given img_id 
    '''
    # omit file extension
    if '.' in img_id:
        img_id = '.'.join(img_id.split('.')[:-1])

    composition = request.args.get("composition", "base").split(",")
    force_generate = request.args.get("force_generate", False, bool)
    formID = request.args.get("formID", "", str)
    if force_generate and formID in used_formIDs:
        force_generate = False
    ensure_archive_video(img_id, emotion, version,
                         force_generate=force_generate, composition=composition)

    used_formIDs.add(formID)

    return send_from_directory("../output/archive", f"{img_id}_{emotion}_{version}.mp4")


@app.route("/output/<string:img_id>/<string:emotion>", methods=["POST"])
def change_output_version(img_id, emotion):
    # omit file extension
    if '.' in img_id:
        img_id = '.'.join(img_id.split('.')[:-1])

    # get request version from request
    body_json = request.get_json(force=True)
    if "version" not in body_json:
        raise_error_response("expected version in body", status=400)
    version = body_json["version"]

    set_main_version(img_id, emotion, version)

    return json.dumps({"response": "success"})


@app.route("/images/detect", methods=["POST"])
def detect_image():
    img = get_request_image("file")
    r = detect(img,
               skip_images=request.args.get("skip_images", "").split(","),
               good_match_threshold=request.args.get(
                   "good_match_threshold", 20, type=int)
               )
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
        if not os.path.exists("output/archive"):
            os.makedirs("output/archive")
        if not os.path.exists("images"):
            os.makedirs("images")
        if not os.path.exists("images/processing"):
            os.makedirs("images/processing")
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
