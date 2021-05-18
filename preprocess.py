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
    draw_points(vid.get_frame(0)[0].img,coords, (int(vid.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
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
            cv2.circle(i, (point[0], point[1]), radius=2,color=(0,0,255), thickness=-1)
        out.write(i)
    out.release()

if __name__ == "__main__":
    if not os.path.exists("preprocess"):
        try: 
            os.makedirs("preprocess")
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # video_file_name= "katja_scared.mp4" 

    # preprocess_video(video_file_name, FPS)

    # to_img = Image(cv2.imread("images/image13.jpg"))
    # generate_video("katja_scared", to_img, "image13")

   
    # emojis = ["inputAnger3", "inputHappiness", "tajda_sad", "katja_sleepy", "inputSmirk", "tajda_flirty", "katja_suspicious"]
    # imgs = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg",
    # "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg", "image11.jpg", "image12.jpg",
    # "image13.jpg", "image14.jpg", "image15.jpg"]

    # imgs = ["image1.jpg", "image2.jpg", "image4.jpg", "image5.jpg", "image6.jpg",         #leva smirk
    # "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg", "image11.jpg",
    # "image15.jpg"]

    # imgs = ["image3.jpg", "image12.jpg"]      #desna smirk

    # for img in imgs:
    #     to_img = Image(cv2.imread("images/" + img))
    #     for emoji in emojis:
    #         img_n = img.split('.')
    #         generate_video(emoji, to_img, img_n[0])
