import json
import os

_landmarks = os.listdir("landmarks")
LANDMARKS = list(map(lambda x: x.split(".")[0], _landmarks))


class Landmarks:
    def __init__(self, name: str):
        self.name = name
        json_path = "landmarks/" + name + ".json"
        with open(json_path) as json_file:
            data = json.load(json_file)
            self.type = name.split("_")[0]
            self.fps = data["fps"]
            self.frames = data["coords"]
            self.faces = data["face"]


def generator_all_landmarks():
    for landmark_name in LANDMARKS:
        yield Landmarks(landmark_name)


def list_landmarks():
    '''
    list_landmarks returns a list of all landmarks in the landmarks folder with no file contents
    '''
    def landmark(name):
        '''
        return landmark info based on name
        '''
        split = name.split("_")
        type = split[0]
        version = split[1]
        return {"type": type, "version": version, "name": name}

    return {l["name"]: l for l in map(landmark, LANDMARKS)}
