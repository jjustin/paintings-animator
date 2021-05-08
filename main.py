import cv2
from scipy.spatial import distance as dist
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from tqdm import tqdm
from math import floor
import json, os, copy, errno
from uuid import uuid4
from threading import Thread

from flask import Flask, send_from_directory, request, Response
app = Flask(__name__)


N_OF_LANDMARKS = 68
MOUTH_AR_THRESH = 0.79
FPS = 6
SAFE_BORDER_SCALE=1.2
VIDEOS = ["input", "input2"]

predictor = dlib.shape_predictor(
    f"shape_predictor_{N_OF_LANDMARKS}_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


class Image:
    def __init__(self, img):
        # cv2 image object
        self.img = img
        # gray representation
        self.gray = cv2.cvtColor(src=self.img, code=cv2.COLOR_BGR2GRAY)
        faces = detector(self.gray)
        self.contains_face = len(faces) != 0
        if(self.contains_face):
            # face's data
            self.face = faces[0]
            self.landmarks = predictor(image=self.gray, box=self.face)
            self.rows, self.cols, self.ch = img.shape
            self.points = [[self.landmarks.part(
                i).x, self.landmarks.part(i).y]for i in range(0, N_OF_LANDMARKS)]
            self.border = [[0, 0], [self.cols, 0],
                           [0, self.rows], [self.cols, self.rows]]

    def size(self):
        return (self.cols, self.rows)

    # draws face detection data on img
    def draw(self, points=None, face=None):
        if points is None:
            points = self.points
        if face is None:
            face = self.face
        img = self.img.copy()
        cv2.rectangle(img=img, pt1=(face.left(), face.top()), pt2=(
            face.right(), face.bottom()), color=(0, 0, 255), thickness=4)
        for [x, y] in points:
            # Draw a circle
            cv2.circle(img=img, center=(floor(x), floor(y)), radius=3,
                       color=(0, 255, 0), thickness=-1)
    # [:,:,::-1] converts to RGB color space
        return img

    def add_text(self, text, font=cv2.FONT_HERSHEY_SIMPLEX, bottomLeftCornerOfText=(10, 10), fontScale=1, fontColor=(255, 255, 255), lineType=2):
        img = self.img.copy()
        cv2.putText(img, text, bottomLeftCornerOfText,
                    font, fontScale, fontColor, lineType)
        return img

    # applies from_img's face landmarks to image
    def apply(self, anchor_points, from_img_points, from_img_face_points, unsafe_border):
        if not self.contains_face:
            return self.add_text("No face found", fontColor=(255, 0, 0))

        img = self.img.copy()

        # face's center
        fx1, fy1 = from_img_points[27][0], from_img_points[27][1]
        ax1, ay1 = anchor_points[27][0], anchor_points[27][1]

        # look at what changed in picture
        changes = [[(from_img_points[i][0] - fx1) + ax1 - anchor_points[i][0],
                    (from_img_points[i][1] - fy1) + ay1 - anchor_points[i][1]]
                   for i in range(N_OF_LANDMARKS)]

        from_pts = copy.deepcopy(from_img_points)

        # # handle mouth
        A = dist.euclidean(from_pts[51], from_pts[59])
        B = dist.euclidean(from_pts[53], from_pts[57])
        C = dist.euclidean(from_pts[49], from_pts[55])

        # # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # # face's coordinates
        tx1, ty1 = self.face.left(), self.face.top()
        tx2, ty2 = self.face.right(), self.face.bottom()
        fx1, fy1 = from_img_face_points[0], from_img_face_points[2]
        fx2, fy2 = from_img_face_points[1], from_img_face_points[3]

        x_scale = (tx2-tx1)/(fx2-fx1)
        y_scale = (ty2-ty1)/(fy2-fy1)

        for i in range(N_OF_LANDMARKS):
            from_pts[i][0] = self.points[i][0] + x_scale*changes[i][0]  # x
            from_pts[i][1] = self.points[i][1] + y_scale*changes[i][1]  # y
        border_points = unsafe_border.to_rect_points()
        tform = PiecewiseAffineTransform()
        tform.estimate(np.float32(from_pts+border_points+self.border),  # self.border has to be self, so that image does not compress
                       np.float32(self.points+border_points+self.border))

        img = warp(img, tform, output_shape=(self.rows, self.cols))
        
        # overwrite mouth
        if(mar > MOUTH_AR_THRESH):
            pts = np.int32(from_pts[60:68])
            cv2.fillPoly(img, [pts], (255, 255, 255))
       
        return img

class Video:
    def __init__(self, vid):
        self.vid = vid

    def get_frame(self, ms):
        self.vid.set(cv2.CAP_PROP_POS_MSEC, ms)
        hasFrame, img = self.vid.read()
        return Image(img), hasFrame

    def length(self):   # in seconds
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        totalNoFrames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        return float(totalNoFrames) / float(fps)

class UnsafeBorder:
    def __init__(self, top, bottom, left, right):
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right

    def to_rect_points(self):
        return [
            [self._left, self._top],
            [self._right, self._top],
            [self._left, self._bottom],
            [self._right, self._bottom]
        ]
    def top(self): 
        return self._top
    def bottom(self): 
        return self._bottom
    def left(self): 
        return self._left
    def right(self): 
        return self._right

def get_unsafe_border(frames, to_img: Image):
    # get info on video data
    top, bottom, left, right = offset_from_anchor_point(frames)
    anchor_top, anchor_bottom, anchor_left, anchor_right = offset_from_anchor_point([frames[0]])
    from_top, from_bottom, from_left, from_right = offset_from_anchor_point([to_img.points])
    
    top_scale = (from_top)/(anchor_top)
    bottom_scale = (from_bottom)/(anchor_bottom)
    left_scale = (from_left)/(anchor_left)
    right_scale = (from_right)/(anchor_right)

    return UnsafeBorder(
     int(to_img.points[27][1] - top * top_scale *SAFE_BORDER_SCALE),
     int(to_img.points[27][1] - bottom * bottom_scale *SAFE_BORDER_SCALE), 
     int(to_img.points[27][0] - left * left_scale *SAFE_BORDER_SCALE), 
     int(to_img.points[27][0] - right * right_scale *SAFE_BORDER_SCALE))

def offset_from_anchor_point(frames):
    x_diff_left = float('-inf')
    x_diff_right = float('inf')
    y_diff_up = float('-inf')
    y_diff_down = float('inf')

    # look over all frames, to find the highest and lowest points, that are still face affected
    for frame in frames:
        for point in frame:
            xdiff = frame[27][0] - point[0]
            ydiff = frame[27][1] - point[1]
            if(x_diff_left < xdiff):
                x_diff_left = xdiff
            if(x_diff_right > xdiff):
                x_diff_right = xdiff
            if(y_diff_up < ydiff):
                y_diff_up = ydiff
            if(y_diff_down > ydiff):
                y_diff_down = ydiff
    return (y_diff_up, y_diff_down,x_diff_left, x_diff_right)

@app.route("/images", methods=['GET'])
def getImages():
    images = os.listdir("images")
    images.remove("processing")
    return json.dumps({"images": images, "processing": os.listdir("images/processing")})

@app.route("/<path:path>", methods=['GET'])
def getStatic(path):  
    return send_from_directory('.', path)

@app.route("/addImage", methods=['PUT'])
def add_image():
    if 'file' not in request.files:
        print('No file part')
        return Response(json.dumps({"error": "no file uploaded"}), status=400)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file:
        filename = str(uuid4()) + "." +file.filename.split(".")[-1]
        filepath = "images/processing/"+filename
        file.save(filepath)

        to_img = Image(cv2.imread(filepath))
        if not to_img.contains_face:
            os.remove(filepath)
            return {"error": "no face detected"}
        Thread(target=handle_image_upload, args=[to_img, filename]).start()
        return {"response": "processing"}

    print('File is empty')
    return json.dumps({"error":"file is empty"})

def handle_image_upload(to_img, img_name):
    generate_all_videos(to_img, img_name)
    os.rename(f"images/processing/{img_name}", f"images/{img_name}")

def generate_all_videos(to_img, img_name):
    print("Starting video processing")
    for video in VIDEOS:
        Thread(target=generate_video, args=[video, to_img, img_name]).start()

def generate_video(video_n, to_img, img_name):
    print(f"Generating video from image {img_name} and video {video_n}")

    # create output object
    img_n  = img_name.split('.')
    output_name = './output/output_' + video_n + '_' + img_n[0] + '.mp4'
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(*"mp4v"), FPS, to_img.size())
 
    #get coordinates for every frame
    json_path = './preprocess/preprocess_' + video_n + '.json'
    with open(json_path) as json_file:
        data = json.load(json_file)

    frames, face_frames = data["coords"], data["face"]
    anchor_frames = copy.deepcopy(frames[0])

    unsafe_border = get_unsafe_border(frames, to_img)

    for i in tqdm(range(len(frames))):
        frame_img = to_img.apply(anchor_frames, frames[i], face_frames[i], unsafe_border) 
        frame_img = (frame_img*255).astype(np.uint8)   # image depth set
        out.write(frame_img)
    out.release()
    print("done")

if __name__ == "__main__":
    # Create required dirs if they do not exist
    if not os.path.exists("output"):
        try: 
            os.makedirs("output")
            os.makedirs("images")
            os.makedirs("images/processing")
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    app.run(host="0.0.0.0", port=5000, debug=True)
