from .image import Image
import cv2
import json

# TODO: Automize
LANDMARKS = [
    "tajda_sad",
    "katja_happy",
    "katja_sleepy",
    "tajda_flirty",
    "katja_suspicious",
]


class Landmarks:
    def __init__(self, name: str):
        self.name = name
        json_path = "preprocess/preprocess_" + name + ".json"
        with open(json_path) as json_file:
            data = json.load(json_file)
            self.fps = data["fps"]
            self.frames = data["coords"]
            self.faces = data["face"]


def get_all_landmarks():
    for landmark_name in LANDMARKS:
        yield Landmarks(landmark_name)
