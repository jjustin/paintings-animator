import cv2
import numpy as np
from tqdm import tqdm
import copy
from storage.image import Image, get_unsafe_border
from storage.landmark import Landmarks, generator_all_landmarks


def generate_all_videos(to_img, img_name):
    print("Starting video processing")
    # threads = []
    for landmarks in generator_all_landmarks():
        generate_video(landmarks, to_img, img_name)


def generate_video(landmarks: Landmarks, to_img: Image, img_name: str):
    print(
        f"Generating video from image {img_name} and landmarks {landmarks.name}")

    # create output object
    img_n = img_name.split(".")
    output_name = "output/output_" + landmarks.name + "_" + img_n[0] + ".mp4"
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(*"avc1"), landmarks.fps, to_img.size())

    anchor_frames = copy.deepcopy(landmarks.frames[0])

    unsafe_border = get_unsafe_border(landmarks.frames, to_img)

    for i in tqdm(range(len(landmarks.frames)), landmarks.name):
        frame_img = to_img.apply(
            anchor_frames, landmarks.frames[i], landmarks.faces[i], unsafe_border, draw_overlay=False
        )
        frame_img = (frame_img).astype(np.uint8)  # image depth set
        out.write(frame_img)
    out.release()
    print("done")
