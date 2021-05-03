from main import Image, Video, FPS
import cv2
from tqdm import tqdm
from math import floor
import json,os, errno

def preprocess_video(new_video, fps):
    #read the video
    print("loading the video") 
    vid = Video(cv2.VideoCapture(new_video))
    # anchor, _ = vid.get_frame(0)                                              #first frame
    print("done loading the video")

    coords = []
    face_coords = []
    vid_len = vid.length()   
    for frame in tqdm(range(floor(vid_len*fps))):
        from_img, has_frame = vid.get_frame(1/fps*frame*1000)

        if (has_frame):
            coords.append(from_img.points)
            #saves face coords (rectangle) from one frame into a list, L R T B
            fc_left = from_img.face.left()
            fc_right = from_img.face.right()
            fc_top = from_img.face.top()
            fc_bottom = from_img.face.bottom()
            fc_list = [fc_left, fc_right, fc_top, fc_bottom]
            face_coords.append(fc_list)
        else:
            coords.append()

    obj = {"coords":coords, "face":face_coords}

    video_n = new_video.split('.')
    file_path = './preprocess/preprocess_' + video_n[0] + '.json'
    with open(file_path, 'w') as out_file:
        json.dump(obj, out_file)
        
if __name__ == "__main__":
    if not os.path.exists("preprocess"):
        try: 
            os.makedirs("preprocess")
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    video_file_name= "input.mov" 

    preprocess_video(video_file_name, FPS)

