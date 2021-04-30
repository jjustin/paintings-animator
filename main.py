import cv2
from scipy.spatial import distance as dist
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from tqdm import tqdm
from math import floor
import json
import copy

N_OF_LANDMARKS = 68
MOUTH_AR_THRESH = 0.79

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
            self.containsFace = True
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
    def apply(self, anchor_points, from_img_points, from_img_face_points):
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

        tform = PiecewiseAffineTransform()
        tform.estimate(np.float32(from_pts+self.border),  # border has to be self, so that image does not compress
                       np.float32(self.points + self.border))

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


def preprocess_video(new_video, fps):
    #read the video
    print("loading the video")
    vid = Video(cv2.VideoCapture(new_video))
    # anchor, _ = vid.get_frame(0)                                              #first frame
    print("done loading the video")

    coords_dict = {}
    face_coords_dict = {}
    vid_len = vid.length()   
    for frame in tqdm(range(floor(vid_len*fps))):
        from_img, has_frame = vid.get_frame(1/fps*frame*1000)

        if (has_frame):
            coords_dict[frame] = from_img.points
            #saves face coords (rectangle) from one frame into a list, L R T B
            fc_left = from_img.face.left()
            fc_right = from_img.face.right()
            fc_top = from_img.face.top()
            fc_bottom = from_img.face.bottom()
            fc_list = [fc_left, fc_right, fc_top, fc_bottom]
            face_coords_dict[frame] = fc_list

    obj = [coords_dict, face_coords_dict]

    video_n = new_video.split('.')
    file_path = './preprocess/preprocess_' + video_n[0] + '.txt'
    with open(file_path, 'w') as out_file:
        json.dump(obj, out_file)


if __name__ == "__main__":
    # read the image
    print("loading the image") 
    #change image name here
    img_name = "to4.jpeg"                                
    to_img = Image(cv2.imread(img_name))
    print("done loading the image")

    fps = 6
    #change the name for new_video for every new video
    new_video = "input.mov"

    img_n  = img_name.split('.')
    video_n = new_video.split('.')

    output_name = 'output_' + video_n[0] + '_' + img_n[0] + '.mp4'
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, to_img.size())

    # preprocess video, uncomment for preprocessing new video
    # preprocess_video(new_video, fps)

    #get coordinates for every frame
    file_path = './preprocess/preprocess_' + video_n[0] + '.txt'
    with open(file_path) as json_file:
        data = json.load(json_file)

    points, face_points = data[0], data[1]
    anchor_points = copy.deepcopy(points['0'])
    
    for i in points:
        frame_img = to_img.apply(anchor_points, points[i], face_points[i])
        frame_img = (frame_img*255).astype(np.uint8)   # image depth set
        out.write(frame_img)
    out.release()

