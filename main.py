import cv2
from scipy.spatial import distance as dist
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from tqdm import tqdm
from math import floor
import json, os, copy

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

if __name__ == "__main__":
    if not os.path.exists("output"):
        try: 
            os.makedirs("output")
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


    # read the image
    print("loading the image") 
    #change image name here
    img_name = "to4.jpg"                                
    to_img = Image(cv2.imread(img_name))
    print("done loading the image")

    fps = 6
    video_n = "input"

    img_n  = img_name.split('.')
    output_name = './output/output_' + video_n + '_' + img_n[0] + '.mp4'
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, to_img.size())
 
    #get coordinates for every frame
    json_path = './preprocess/preprocess_' + video_n + '.json'
    with open(json_path) as json_file:
        data = json.load(json_file)

    points, face_points = data["coords"], data["face"]
    anchor_points = copy.deepcopy(points[0])
    
    for i in range(len(points)):
        frame_img = to_img.apply(anchor_points, points[i], face_points[i])
        frame_img = (frame_img*255).astype(np.uint8)   # image depth set
        out.write(frame_img)
    out.release() 
    print("done")
