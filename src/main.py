import cv2
from math import sqrt
import dlib
import numpy as np
# from skimage.transform import PiecewiseAffineTransform, warp
from piecewiseAffine import Transformer, has_cuda
from tqdm import tqdm
from math import floor
import json
import os
import copy
import errno
from uuid import uuid4
from threading import Thread

from helpers import Timer

from flask import Flask, send_from_directory, request, Response

app = Flask(__name__)


N_OF_LANDMARKS = 68
MOUTH_AR_THRESH = 0.79
SAFE_BORDER_SCALE = 1.05
CENTER_POINT_IX = 27
VIDEOS = [
    "tajda_sad",
    "katja_happy",
    "katja_sleepy",
    "tajda_flirty",
    "katja_suspicious",
]

predictor = dlib.shape_predictor(
    f"shape_predictor_{N_OF_LANDMARKS}_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


initTimer = Timer("init")
applyTimer = Timer("apply")
totalTimer = Timer("total")


class Image:
    def __init__(self, img):
        initTimer.start()
        # cv2 image object
        self.img = img
        if has_cuda:
            self.img = cv2.cuda_GpuMat()
            self.img.upload(img)
        # gray representation
        self.gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = detector(self.gray)
        self.contains_face = len(faces) != 0
        if self.contains_face:
            # face's data
            self.face = faces[0]
            self.landmarks = predictor(image=self.gray, box=self.face)
            self.rows, self.cols, self.ch = img.shape
            self.points = [
                [self.landmarks.part(i).x, self.landmarks.part(i).y]
                for i in range(0, N_OF_LANDMARKS)
            ]
            self.border = [
                [0, 0],
                [self.cols, 0],
                [0, self.rows],
                [self.cols, self.rows],
            ]
        initTimer.end()

    def size(self):
        return (self.cols, self.rows)

    # draws face detection data on img
    def draw(self, points=None, face=None):
        if points is None:
            points = self.points
        if face is None:
            face = self.face
        img = self.img.copy()
        cv2.rectangle(
            img=img,
            pt1=(face.left(), face.top()),
            pt2=(face.right(), face.bottom()),
            color=(0, 0, 255),
            thickness=4,
        )
        for [x, y] in points:
            # Draw a circle
            cv2.circle(
                img=img,
                center=(floor(x), floor(y)),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )
        # [:,:,::-1] converts to RGB color space
        return img

    def add_text(
        self,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        bottomLeftCornerOfText=(10, 10),
        fontScale=1,
        fontColor=(255, 255, 255),
        lineType=2,
    ):
        img = self.img.copy()
        cv2.putText(
            img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType
        )
        return img

    # applies from_img's face landmarks to image
    def apply(
        self,
        anchor_points,
        from_img_points,
        from_img_face_points,
        unsafe_border,
        draw_overlay=False,
    ):
        applyTimer.start()
        if not self.contains_face:
            return self.add_text("No face found", fontColor=(255, 0, 0))

        # img = self.img.copy()

        # face's center
        fx1, fy1 = (
            from_img_points[CENTER_POINT_IX][0],
            from_img_points[CENTER_POINT_IX][1],
        )
        ax1, ay1 = anchor_points[CENTER_POINT_IX][0], anchor_points[CENTER_POINT_IX][1]

        # look at what changed in picture
        changes = [
            [
                (from_img_points[i][0] - fx1) + ax1 - anchor_points[i][0],
                (from_img_points[i][1] - fy1) + ay1 - anchor_points[i][1],
            ]
            for i in range(N_OF_LANDMARKS)
        ]

        from_pts = copy.deepcopy(from_img_points)

        # # handle mouth
        A = euclidean_dist(from_pts[51], from_pts[59])
        B = euclidean_dist(from_pts[53], from_pts[57])
        C = euclidean_dist(from_pts[49], from_pts[55])

        # # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # # face's coordinates
        tx1, ty1 = self.face.left(), self.face.top()
        tx2, ty2 = self.face.right(), self.face.bottom()
        fx1, fy1 = from_img_face_points[0], from_img_face_points[2]
        fx2, fy2 = from_img_face_points[1], from_img_face_points[3]

        x_scale = (tx2 - tx1) / (fx2 - fx1)
        y_scale = (ty2 - ty1) / (fy2 - fy1)

        for i in range(N_OF_LANDMARKS):
            from_pts[i][0] = self.points[i][0] + x_scale * changes[i][0]  # x
            from_pts[i][1] = self.points[i][1] + y_scale * changes[i][1]  # y

        border_points = unsafe_border.to_points()
        # self.border has to be included, so that image does not collapse onto itself(black border)
        final_points = from_pts + border_points + self.border
        start_points = self.points + border_points + self.border

        transformTimer = Timer("transform").start()
        # t = Timer("estimate").start()
        # tform = PiecewiseAffineTransform()
        # tform.estimate(np.float32(final_points), np.float32(start_points))
        # t.end()

        # t = Timer("warp").start()
        # img = warp(
        #     img,
        #     tform,
        #     output_shape=(self.rows, self.cols),
        #     mode="reflect",
        # )
        # t.end()
        t = Transformer(self.img, start_points, final_points)
        img = t.warp_affine_pw()
        transformTimer.end()

        if draw_overlay:
            subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
            for point in from_pts + border_points:
                x, y = int(point[0]), int(point[1])
                subdiv.insert((x, y))
            draw_delaunay(img, subdiv, (255, 255, 255))

        # overwrite mouth
        if mar > MOUTH_AR_THRESH:
            pts = np.int32(from_pts[60:68])
            cv2.fillPoly(img, [pts], (255, 255, 255))

        applyTimer.end()
        # return cupy.asnumpy(img)
        return img


def euclidean_dist(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rect_contains(rect, point):
    # Check if a point is inside a rectangle
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    size = img.shape
    y = size[0]
    x = size[1]
    r = (0, 0, x, y)
    subdiv.insert((0, 0))
    subdiv.insert((x - 1, 0))
    subdiv.insert((0, y - 1))
    subdiv.insert((x - 1, y - 1))
    triangleList = subdiv.getTriangleList()

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


class Video:
    def __init__(self, vid: cv2.VideoCapture):
        self.vid = vid

    def get_frame(self, ms):
        self.vid.set(cv2.CAP_PROP_POS_MSEC, ms)
        hasFrame, img = self.vid.read()
        return Image(img), hasFrame

    def length(self):  # in seconds
        fps = self.vid.get(cv2.CAP_PROP_FPS)
        totalNoFrames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        return float(totalNoFrames) / float(fps)


class UnsafeBorder:
    def __init__(self, top: int, bottom, left, right):
        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right

    def to_points(self):
        return [
            [self._left, self._top],
            [self._right, self._top],
            [self._left, self._bottom],
            [self._right, self._bottom],
            [(self._right + self._left) // 2, self._top],
            [(self._right + self._left) // 2, self._bottom],
            [self._right, (self._bottom + self._top) // 2],
            [self._left, (self._bottom + self._top) // 2],
        ]


def get_unsafe_border(frames, to_img: Image):
    # get info on video data
    top, bottom, left, right = offset_from_anchor_point(frames)
    anchor_top, anchor_bottom, anchor_left, anchor_right = offset_from_anchor_point(
        [frames[0]]
    )
    from_top, from_bottom, from_left, from_right = offset_from_anchor_point(
        [to_img.points]
    )

    top_scale = (from_top) / (anchor_top)
    bottom_scale = (from_bottom) / (anchor_bottom)
    left_scale = (from_left) / (anchor_left)
    right_scale = (from_right) / (anchor_right)

    return UnsafeBorder(
        int(to_img.points[CENTER_POINT_IX][1] -
            top * top_scale * SAFE_BORDER_SCALE),
        int(
            to_img.points[CENTER_POINT_IX][1]
            - bottom * bottom_scale * SAFE_BORDER_SCALE
        ),
        int(to_img.points[CENTER_POINT_IX][0] -
            left * left_scale * SAFE_BORDER_SCALE),
        int(
            to_img.points[CENTER_POINT_IX][0] -
            right * right_scale * SAFE_BORDER_SCALE
        ),
    )


def offset_from_anchor_point(frames):
    x_diff_left = float("-inf")
    x_diff_right = float("inf")
    y_diff_up = float("-inf")
    y_diff_down = float("inf")

    # look over all frames, to find the highest and lowest points, that are still face affected
    for frame in frames:
        for point in frame:
            xdiff = frame[CENTER_POINT_IX][0] - point[0]
            ydiff = frame[CENTER_POINT_IX][1] - point[1]
            if x_diff_left < xdiff:
                x_diff_left = xdiff
            if x_diff_right > xdiff:
                x_diff_right = xdiff
            if y_diff_up < ydiff:
                y_diff_up = ydiff
            if y_diff_down > ydiff:
                y_diff_down = ydiff
    return (y_diff_up, y_diff_down, x_diff_left, x_diff_right)


@app.route("/images", methods=["GET"])
def getImages():
    images = os.listdir("images")

    if "processing" in images:
        images.remove("processing")
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    return json.dumps({"images": images, "processing": os.listdir("images/processing")})


@app.route("/", defaults={"path": "src/index_additive.html"}, methods=["GET"])
@app.route("/<path:path>", methods=["GET"])
def getStatic(path):
    return send_from_directory("../", path)


@app.route("/exists/<path:image>", methods=["GET"])
def getExists(image):
    return json.dumps(os.path.isfile("images/" + image))


@app.route("/addImage", methods=["POST"])
def add_image():
    if "file" not in request.files:
        print("No file part")
        return Response(json.dumps({"error": "no file uploaded"}), status=400)
    file = request.files["file"]

    if file:
        filename = str(uuid4()) + "." + file.filename.split(".")[-1]
        filepath = "images/processing/" + filename
        file.save(filepath)

        to_img = Image(cv2.imread(filepath))
        if not to_img.contains_face:
            os.remove(filepath)
            return {"error": "no face detected"}
        Thread(target=handle_image_upload, args=[to_img, filename]).start()
        return {"response": "processing", "img_name": filename}

    print("File is empty")
    return json.dumps({"error": "file is empty"})


@app.route("/times", methods=["GET"])
def get_times():
    return Timer.stat()


def handle_image_upload(to_img, img_name):
    totalTimer.start()
    generate_all_videos(to_img, img_name)
    totalTimer.end()
    os.rename(f"images/processing/{img_name}", f"images/{img_name}")


def generate_all_videos(to_img, img_name):
    print("Starting video processing")
    # threads = []
    for video in VIDEOS:
        #     t = Thread(target=generate_video, args=[video, to_img, img_name])
        #     t.start()
        #     threads.append(t)
        # for t in threads:
        #     t.join()
        generate_video(video, to_img, img_name)


def generate_video(video_n, to_img: Image, img_name):
    print(f"Generating video from image {img_name} and video {video_n}")

    # get coordinates for every frame
    json_path = "preprocess/preprocess_" + video_n + ".json"
    with open(json_path) as json_file:
        data = json.load(json_file)

    # create output object
    img_n = img_name.split(".")
    output_name = "output/output_" + video_n + "_" + img_n[0] + ".mp4"
    out = cv2.VideoWriter(
        output_name, cv2.VideoWriter_fourcc(
            *"avc1"), data["fps"], to_img.size()
    )

    frames, face_frames = data["coords"], data["face"]
    anchor_frames = copy.deepcopy(frames[0])

    unsafe_border = get_unsafe_border(frames, to_img)

    for i in tqdm(range(len(frames)), video_n):
        frame_img = to_img.apply(
            anchor_frames, frames[i], face_frames[i], unsafe_border, draw_overlay=False
        )
        frame_img = (frame_img).astype(np.uint8)  # image depth set
        out.write(frame_img)
    out.release()
    print("done")


if __name__ == "__main__":
    # Create required dirs if they do not exist
    try:
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("images"):
            os.makedirs("images")
        if not os.path.exists("images/processing"):
            os.makedirs("images/processing")
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

    app.run(host="0.0.0.0", port=5000, debug=True)
