from typing import List
from storage.video import Video
from storage.image import Image

import cv2
from tqdm import tqdm
from math import floor
import json, os, errno



def preprocess_video(new_video):
    # read the video
    print(f"loading {new_video}")
    vid = Video(video=cv2.VideoCapture(new_video))
    # anchor, _ = vid.get_frame(0)                                              #first frame
    print("loaded")

    coords = []
    face_coords = []
    vid_len = vid.length()
    fps = vid.fps
    images = []
    for frame in tqdm(range(floor(vid_len * fps)), "Read to memory"):
        img, has_frame = vid.get_frame(1 / fps * frame * 1000)
        images.append(Image(img))

    for frame in tqdm(range(floor(vid_len * fps)), "Avg + face"):
        from_img = images[frame]
        if has_frame:
            coords.append(average(images, frame, n = int(fps/6)))

            # saves face coords (rectangle) from one frame into a list, L R T B
            fc_left = from_img.face.left()
            fc_right = from_img.face.right()
            fc_top = from_img.face.top()
            fc_bottom = from_img.face.bottom()
            fc_list = [fc_left, fc_right, fc_top, fc_bottom]
            face_coords.append(fc_list)
        else:
            raise Exception("Face not detected on frame: " + frame)

    obj = {"coords": coords, "face": face_coords, "fps": fps}

    video_n = new_video.split(".")[0]

    # # Creates video output of detected points
    draw_points(
        vid.get_frame(0)[0],
        coords,
        (vid.w,vid.h),
        fps,
        name = video_n
    )

    draw_vectors(
        vid.get_frame(0)[0],
        coords,
        (vid.w,vid.h),
        fps,
        name = video_n
    )

    file_path = "./landmarks/" + video_n + ".json"
    with open(file_path, "w") as out_file:
        json.dump(obj, out_file)


def average(images: List[Image], ix, n=5):
    out = []
    div = 0
    for frame in range(max(0, ix - n + 1), min(ix + n, len(images))):
        div += n - abs(frame - ix)

    for i in range(len(images[0].points)):  # go over each point
        avgx = 0
        avgy = 0
        for frame in range(
            max(0, ix - n + 1), min(ix + n, len(images))
        ):  # find average of last/next few frames
            mult = n - abs(frame - ix)
            avgx += mult * images[frame].points[i][0]
            avgy += mult * images[frame].points[i][1]
        out.append([int(avgx / div), int(avgy / div)])
    return out


def draw_points(bg, frames, size, fps, name:str = ""):
    out = cv2.VideoWriter(f"points_{name}.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, size)

    for frame in frames:
        i = bg.copy()
        cv2.rectangle(i, (0, 0), size, (0, 0, 0), thickness=-1)
        for point in frame:
            cv2.circle(
                i, (point[0], point[1]), radius=2, color=(0, 0, 255), thickness=8
            )
        out.write(i)
    out.release()


def draw_vectors(bg, frames, size, fps, name:str = ""):
    out = cv2.VideoWriter(f"vectors_{name}.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, size)

    for frame in frames:
        i = bg.copy()
        cv2.rectangle(i, (0, 0), size, (0, 0, 0), thickness=-1)
        for j in range(len(frame)):
            cv2.arrowedLine(
                i,
                (frame[j][0], frame[j][1]),
                (frames[0][j][0], frames[0][j][1]),
                color=(0, 0, 255),
                thickness=5,
            )
            cv2.circle(
                i, (frame[j][0], frame[j][1]), radius=2, color=(0, 0, 255), thickness=8
            )
            cv2.circle(
                i,
                (frames[0][j][0], frames[0][j][1]),
                radius=2,
                color=(0, 255, 0),
                thickness=8,
            )
        out.write(i)
    out.release()


if __name__ == "__main__":
    if not os.path.exists("landmarks"):
        try:
            os.makedirs("landmarks")
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    from sys import argv
    if len(argv) <=1:
        raise

    for video_file_name in argv[1:]:
        preprocess_video(video_file_name)
