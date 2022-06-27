from sys import flags
from time import sleep
import cv2

from flask import Response

class Video:
    def __init__(self,  video: cv2.VideoCapture):
        self.video = video

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @classmethod
    def from_img_id(cls, img_id: str, emotion: str, version: str = None):
        '''
        init video from img_id and emotion and optional version
        '''
        if version is None:
            video = cv2.VideoCapture(f"output/{img_id}_{emotion}.mp4")
        else:
            video = cv2.VideoCapture(
                f"output/archive/{img_id}_{emotion}_{version}.mp4")
        return cls(video)

    def __del__(self):
        #releasing video
        self.video.release()

    def get_frame(self, ms):
        self.video.set(cv2.CAP_PROP_POS_MSEC, ms)
        hasFrame, img = self.video.read()
        return img, hasFrame

    def length(self):  # in seconds
        return float(self.frame_count) / float(self.fps)
