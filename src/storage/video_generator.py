import cv2
import os
from shutil import copyfile
import numpy as np
from tqdm import tqdm
import copy
from storage.image import Image, get_unsafe_border, read_cv2_image
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


def generate_one_per_emotion(img_id: str, img: Image = None):
    '''
    generate one video for each emotion for given image
    '''
    meta = dict_landmarks_meta()
    for emotion in meta:
        for version in meta[emotion]:
            set_main_version(img_id, emotion, version, img=img)
            break


def set_main_version(img_id: str, emotion: str, version: str, img: Image = None):
    '''
    Sets main version of emotion for given image. Video is generated if it does not exist.

    img - optional image object, if not provided, it will be read from disk
    '''
    ensure_archive_video(img_id, emotion, version, img=img)
    copy_archive_to_main(img_id, emotion, version)


def copy_archive_to_main(img_id: str, emotion: str, version: str):
    '''
    Copy archive video to main video
    '''
    archive_path = f"output/archive/{img_id}_{emotion}_{version}.mp4"
    main_path = f"output/{img_id}_{emotion}.mp4"
    copyfile(archive_path, main_path)


def ensure_archive_video(img_id: str, emotion: str, version: str, img: Image = None) -> str:
    """
    Ensure that the video archive exists.

    img_id - image name
    emotion - emotion name
    version - version of emotion
    img - optional image object, if not provided, it will be read from disk
    """
    landmarks = get_landmarks(emotion, version)
    path = f"output/archive/{img_id}_{emotion}_{version}.mp4"
    _ensure_video(img_id, landmarks, path, img=img)


def _ensure_video(img_id: str, landmarks: Landmarks, path: str, img: Image = None):
    if not os.path.exists(path):
        if img is None:
            img = Image(read_cv2_image(img_id))
        _generate_video(landmarks, img, img_id, path)


def _generate_video(landmarks: Landmarks, to_img: Image, img_id: str, output_path: str):
    print(
        f"Generating video from image {img_id} and landmarks {landmarks.name}")

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"avc1"), landmarks.fps, to_img.size())

    anchor_frames = copy.deepcopy(landmarks.frames[0])

    unsafe_border = get_unsafe_border(landmarks.frames, to_img)

    for i in tqdm(range(len(landmarks.frames)), landmarks.name):
        frame_img = to_img.apply(
            anchor_frames, landmarks.frames[i], landmarks.faces[i], unsafe_border, draw_overlay=False
        )
        frame_img = (frame_img).astype(np.uint8)  # image depth set
        out.write(frame_img)
    out.release()

    print(f"{output_path} generated")
