from main import Image, Video, generate_video
import cv2
from tqdm import tqdm
from math import floor
import json,os, errno

import numpy as np

FPS = 30

def preprocess_video(new_video, fps):
    #read the video
    print("loading the video") 
    vid = Video(cv2.VideoCapture(new_video))
    # anchor, _ = vid.get_frame(0)                                              #first frame
    print("done loading the video")

    coords = []
    face_coords = []
    vid_len = vid.length()
    images = []
    for frame in tqdm(range(floor(vid_len*fps))):
        img, has_frame = vid.get_frame(1/fps*frame*1000)
        images.append(img)

    for frame in tqdm(range(floor(vid_len*fps))):
        from_img = images[frame]
        if (has_frame):
            coords.append(average(images, frame))
            
            #saves face coords (rectangle) from one frame into a list, L R T B
            fc_left = from_img.face.left()
            fc_right = from_img.face.right()
            fc_top = from_img.face.top()
            fc_bottom = from_img.face.bottom()
            fc_list = [fc_left, fc_right, fc_top, fc_bottom]
            face_coords.append(fc_list)
        else:
            raise Exception("Face not detected on frame: " + frame)

    obj = {"coords":coords, "face":face_coords, "fps": fps}
    
    # # Creates video output of detected points
    draw_points(vid.get_frame(0)[0].img, coords, (int(vid.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    draw_vectors(vid.get_frame(0)[0].img, coords, (int(vid.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    video_n = new_video.split('.')
    file_path = './preprocess/preprocess_' + video_n[0] + '.json'
    with open(file_path, 'w') as out_file:
        json.dump(obj, out_file)

def average(images, ix):
    n = 5
    out = []
    div = 0
    for frame in range(max(0, ix-n+1), min(ix+n,len(images))): 
        div += n-abs(frame-ix)

    for i in range(len(images[0].points)): # go over each point
        avgx = 0
        avgy = 0
        for frame in range(max(0, ix-n+1), min(ix+n,len(images))): # find average of last/next few frames
            mult = n-abs(frame-ix)
            avgx += mult*images[frame].points[i][0]
            avgy += mult*images[frame].points[i][1]
        out.append([int(avgx/div), int(avgy/div)])
    return out
            

def draw_points(bg, frames, size):
    out = cv2.VideoWriter("points.mp4", cv2.VideoWriter_fourcc(*"avc1"), FPS, size)

    for frame in frames:
        i = bg.copy()
        cv2.rectangle(i, (0,0), size, (0,0,0), thickness=-1)
        for point in frame:
            cv2.circle(i, (point[0], point[1]), radius=2,color=(0,0,255), thickness=8)
        out.write(i)
    out.release()

def draw_vectors(bg, frames, size):
    out = cv2.VideoWriter("vectors.mp4", cv2.VideoWriter_fourcc(*"avc1"), FPS, size)

    for frame in frames:
        i = bg.copy()
        cv2.rectangle(i, (0,0), size, (0,0,0), thickness=-1)
        for j  in range(len(frame)):
            cv2.arrowedLine(i, (frame[j][0], frame[j][1]), (frames[0][j][0], frames[0][j][1]), color=(0,0,255), thickness=5)
            cv2.circle(i, (frame[j][0], frame[j][1]), radius=2,color=(0,0,255), thickness=8)
            cv2.circle(i, (frames[0][j][0], frames[0][j][1]), radius=2,color=(0,255,0), thickness=8)
        out.write(i)
    out.release()

if __name__ == "__main__":
    if not os.path.exists("preprocess"):
        try: 
            os.makedirs("preprocess")
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    video_file_name= "faceMove.mp4" 

    preprocess_video(video_file_name, FPS)
