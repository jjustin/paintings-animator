import json
import os


class Landmarks:
    def __init__(self, name: str):
        self.name = name
        json_path = "landmarks/" + name + ".json"
        with open(json_path) as json_file:
            data = json.load(json_file)
            self.emotion, self.version = name.split("_")
            self.fps = data["fps"]
            self.frames = data["coords"]
            self.faces = data["face"]


class LandmarksMetadata:
    def __init__(self, filename):
        self.name = filename.split(".")[0]
        self.emotion = self.name.split("_")[0]
        self.version = self.name.split("_")[1]

    def serialize(self):
        return {
            "name": self.name,
            "emotion": self.emotion,
            "version": self.version
        }


def list_landmarks_meta():
    '''
    list_landmarks returns a list of all existing LandmarksMetadata
    '''
    return [LandmarksMetadata(l) for l in os.listdir("landmarks")]


def dict_landmarks_meta():
    '''
    dict_landmarks returns a dictionary of all landmarks' metadatas mapped by type and version
    '''

    out = {}
    for l in list_landmarks_meta():
        if l.emotion not in out:
            out[l.emotion] = {}

        out[l.emotion][l.version] =  l.serialize()

    return out


def get_landmarks(type, version):
    '''
    get_landmarks returns a Landmarks object based on name
    '''
    return Landmarks(get_name(type, version))


def get_name(type, version):
    return f"{type}_{version}"
