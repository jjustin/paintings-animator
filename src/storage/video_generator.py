from typing import List
import cv2
import os
from shutil import copyfile
import numpy as np
from tqdm import tqdm
import copy
from storage.image import Image, FaceBorder, get_unsafe_border, read_cv2_image
from storage.landmark import Landmarks, get_landmarks, dict_landmarks_meta
from helpers import raise_error_response

"""
All generated videos are kept in 'archive' dir, to prevent regenerating them every time they are needed.
This is why, when a video is generated, archive version is created and is later copied in 'output' folder. 
"""


def ensure_video(img_id: str, emotion: str):
    filepath = f"output/{img_id}_{emotion}.mp4"
    if os.path.exists(filepath):
        return

    landmarks = dict_landmarks_meta()
    if emotion not in landmarks:
        raise_error_response(f"No landmarks recorded for: {emotion}")

    for version in landmarks[emotion]:
        set_main_version(img_id, emotion, version)
        break


def generate_one_per_emotion(img_id: str, img: Image = None, force_generate: bool = False):
    '''
    generate one video for each emotion for given image
    '''
    meta = dict_landmarks_meta()
    for emotion in meta:
        for version in meta[emotion]:
            set_main_version(img_id, emotion, version, img, force_generate)
            break


def set_main_version(img_id: str, emotion: str, version: str, img: Image = None, force_generate: bool = False):
    '''
    Sets main version of emotion for given image. Video is generated if it does not exist.

    img - optional image object, if not provided, it will be read from disk
    '''
    ensure_archive_video(img_id, emotion, version, img, force_generate)
    copy_archive_to_main(img_id, emotion, version)


def copy_archive_to_main(img_id: str, emotion: str, version: str):
    '''
    Copy archive video to main video
    '''
    archive_path = f"output/archive/{img_id}_{emotion}_{version}.mp4"
    main_path = f"output/{img_id}_{emotion}.mp4"
    copyfile(archive_path, main_path)


def ensure_archive_video(img_id: str, emotion: str, version: str, img: Image = None, force_generate: bool = False, composition: List[str] = None) -> str:
    """
    Ensure that the video archive exists.

    img_id - image name
    emotion - emotion name
    version - version of emotion
    img - optional image object, if not provided, it will be read from disk
    """
    landmarks = get_landmarks(emotion, version)
    path = f"output/archive/{img_id}_{emotion}_{version}.mp4"
    _ensure_video(img_id, landmarks, path, img, force_generate, composition)


def _ensure_video(img_id: str, landmarks: Landmarks, path: str, img: Image = None, force_generate: bool = False, composition: List[str] = None):
    if not force_generate and os.path.exists(path):
        return

    if img is None:
        img = Image(read_cv2_image(img_id))
    _generate_video(landmarks, img, img_id, path, composition)


def _generate_video(landmarks: Landmarks, to_img: Image, img_id: str, output_path: str, composition: List[str] = None):
    if composition == None:
        composition = ["base"]

    print(
        f"Generating video from image {img_id} and landmarks {landmarks.name} with composition {composition}")

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"avc1"), landmarks.fps, to_img.size())

    unsafe_border = get_unsafe_border(landmarks.frames, to_img)

    composer = OuputComposer(
        to_img, landmarks, unsafe_border, composition, out)

    composer.compose()
    out.release()

    print(f"{output_path} generated")


class OuputComposer:
    def __init__(self, img: Image, landmarks: Landmarks, unsafe_border: FaceBorder, composition: List[str], out: cv2.VideoWriter):
        self.img = img
        self.landmarks = landmarks
        self.unsafe_border = unsafe_border
        self.comp_funcs = []

        for comp in composition:
            if comp == "base":
                self.comp_funcs.append(self._base_animation())

            elif comp.startswith("freeze"):
                l = int(comp.split(":")[1])
                self.comp_funcs.append(self._freeze_animation(l))

            elif comp.startswith("morph_back"):
                l = int(comp.split(":")[1])
                self.comp_funcs.append(self._morph_back_animation(l))

            elif comp.startswith("fade_back"):
                l = int(comp.split(":")[1])
                self.comp_funcs.append(self._fade_back_animation(l))

        self.frames = []
        self.last_frame = img.img_cpu
        self.out = out

        self.lframes = landmarks.frames
        self.last_lframe = landmarks.frames[0]
        self.anchor_lframe = copy.deepcopy(landmarks.frames[0])

    def _add_frame(self, frame, lframe=None):
        if lframe is not None:
            self.last_lframe = lframe

        self.frames.append(frame)
        self.out.write(frame)

        self.last_frame = frame

    '''
    Animations
    '''

    def _base_animation(self):
        '''
        Base animation applies landmarks to image
        '''
        def anim():
            for i in tqdm(range(len(self.lframes)), "Base animation"):
                frame_img = self.img.apply(
                    self.anchor_lframe, self.lframes[i], self.landmarks.faces[i], self.unsafe_border, draw_overlay=False
                )
                self._add_frame(frame_img, lframe=self.lframes[i])
        return anim

    def _freeze_animation(self, length):
        '''
        repeats last frame for `length` frames
        '''
        def anim():
            for _ in tqdm(range(length), "freeze"):
                self._add_frame(self.last_frame)

        return anim

    def _morph_back_animation(self, length):
        '''
        morphs back to original image for `length` frames
        '''
        def anim():
            anchor_face = self.landmarks.faces[0]
            origin_lframe = self.last_lframe
            for i in tqdm(range(length+1), "Fade back"):
                if i == length:
                    self._add_frame(self.img.img_cpu)
                    continue

                lframe = []
                rate = i/length
                for j in range(len(self.anchor_lframe)):
                    x = origin_lframe[j][0] + \
                        ((self.anchor_lframe[j][0] - origin_lframe[j][0])*rate)
                    y = origin_lframe[j][1] + \
                        ((self.anchor_lframe[j][1] - origin_lframe[j][1])*rate)
                    lframe.append([x, y])

                frame_img = self.img.apply(
                    self.anchor_lframe, lframe, anchor_face, self.unsafe_border, draw_overlay=False
                )
                self._add_frame(frame_img, lframe)
        return anim

    def _fade_back_animation(self, length):
        '''
        fades back to original image in span of `length` frames
        '''
        def anim():
            for i in tqdm(range(length), "Fade back"):
                rate = i/length

                frame = (1-rate) * self.img.img_cpu + rate * self.last_frame
                self._add_frame(self.img.img_cpu)
        return anim

    def compose(self):
        for comp_func in self.comp_funcs:
            comp_func()
