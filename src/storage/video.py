import cv2
from .image import Image


class Video:
    def __init__(self, vid: cv2.VideoCapture, name: str):
        self.vid = vid
        self.name = name

    def get_frame(self, ms):
        self.vid.set(cv2.CAP_PROP_POS_MSEC, ms)
        hasFrame, img = self.vid.read()
        return Image(img), hasFrame

    def length(self):  # in seconds
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        totalNoFrames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        return float(totalNoFrames) / float(fps)
