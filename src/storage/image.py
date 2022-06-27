from cmath import pi
import cv2
from math import atan2, sqrt
import dlib
import numpy as np
from helpers import Timer, raise_error_response
from math import floor
import copy
import os
from piecewiseAffine import Transformer, has_cuda


initTimer = Timer("init")
applyTimer = Timer("apply")

N_OF_LANDMARKS = 68
MOUTH_AR_THRESH = 0.79
SAFE_BORDER_SCALE = 1.05
CENTER_POINT_IX = 27  # between the eyes
CENTER_POINT_IX2 = 34  # nose point

PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_FILENAME)
detector = dlib.get_frontal_face_detector()


class UnsafeBorder:
    '''
    UnsafeBorder is used to create a border around the face.
    '''

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


class Image:
    def __init__(self, img, max_height=1200):
        initTimer.start()

        # resize image to prevent long processing times
        if img.shape[0] > max_height:
            scale_factor = max_height / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
            print("input size after resize: ", img.shape)

        # cv2 image object
        self.img_cpu = img
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

            # rotate points to get the face aligned with the y-axis
            dx = self.points[CENTER_POINT_IX2][0] - \
                self.points[CENTER_POINT_IX][0]
            dy = self.points[CENTER_POINT_IX2][1] - \
                self.points[CENTER_POINT_IX][1]

            # x and y switched because we want to rotate to y axis
            angle = 180*atan2(dx, dy)/pi

            self.rotation_matrix = cv2.getRotationMatrix2D(
                self.points[CENTER_POINT_IX], angle=-angle, scale=1)

            self.inverse_rotation_matrix = cv2.getRotationMatrix2D(
                self.points[CENTER_POINT_IX], angle=angle, scale=1)

            self.points = self.rotate_points(self.points)
        initTimer.end()

    def rotate_points(self, points, inverse=False):
        M = self.rotation_matrix
        if inverse:
            M = self.inverse_rotation_matrix

        points = np.array(points, ndmin=2)
        points = np.c_[points, np.ones(points.shape[0])].T
        return (M @ points).T

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

    def apply(
        self,
        landmark_anchor,
        landmark_points,
        landmark_face_border,
        unsafe_border: UnsafeBorder,
        draw_overlay=False,
    ):
        """
        applies from_img's face landmarks to image
        """

        applyTimer.start()
        if not self.contains_face:
            return self.add_text("No face found", fontColor=(255, 0, 0))

        # face's center
        fx1, fy1 = (
            landmark_points[CENTER_POINT_IX][0],
            landmark_points[CENTER_POINT_IX][1],
        )
        ax1, ay1 = landmark_anchor[CENTER_POINT_IX][0], landmark_anchor[CENTER_POINT_IX][1]

        # look at what changed in picture
        changes = [
            [
                (landmark_points[i][0] - fx1) + ax1 - landmark_anchor[i][0],
                (landmark_points[i][1] - fy1) + ay1 - landmark_anchor[i][1],
            ]
            for i in range(N_OF_LANDMARKS)
        ]

        from_pts = copy.deepcopy(landmark_points)

        # # handle mouth
        A = euclidean_dist(from_pts[51], from_pts[59])
        B = euclidean_dist(from_pts[53], from_pts[57])
        C = euclidean_dist(from_pts[49], from_pts[55])

        # # compute the mouth aspect ratio
        mar = (A + B) / (2.0 * C)

        # # face's coordinates
        tx1, ty1 = self.face.left(), self.face.top()
        tx2, ty2 = self.face.right(), self.face.bottom()
        fx1, fy1 = landmark_face_border[0], landmark_face_border[2]
        fx2, fy2 = landmark_face_border[1], landmark_face_border[3]

        x_scale = (tx2 - tx1) / (fx2 - fx1)
        y_scale = (ty2 - ty1) / (fy2 - fy1)

        for i in range(N_OF_LANDMARKS):
            from_pts[i][0] = self.points[i][0] + x_scale * changes[i][0]  # x
            from_pts[i][1] = self.points[i][1] + y_scale * changes[i][1]  # y

        border_points = unsafe_border.to_points()

        final_points = self.rotate_points(np.concatenate(
            (from_pts, border_points)), inverse=True)
        start_points = self.rotate_points(np.concatenate(
            (self.points, border_points)), inverse=True)

        transformTimer = Timer("transform").start()

        t = Transformer(self.img, start_points, final_points)
        img = t.warp_affine_pw()
        transformTimer.end()

        if draw_overlay:
            subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))
            for point in final_points:
                x, y = int(point[0]), int(point[1])
                subdiv.insert((x, y))
            draw_delaunay(img, subdiv, (255, 255, 255))

        # overwrite mouth
        if mar > MOUTH_AR_THRESH:
            pts = np.int32(from_pts[60:68])
            cv2.fillPoly(img, [pts], (255, 255, 255))

        applyTimer.end()

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

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def get_unsafe_border(frames, to_img: Image):
    '''
    Returns UnsafeBorder object that can be applied on to_img to prevent the background from being wiggled.
    '''

    # get offset for input expression frames
    top, bottom, left, right = offset_from_anchor_point(frames)
    # get offset for anchor point, that is used for calculating the scale
    anchor_top, anchor_bottom, anchor_left, anchor_right = offset_from_anchor_point(
        [frames[0]]
    )
    # get offset on destination image
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
    '''
    offset_from_anchor_point is used to get the offset of points from the anchor point.
    offset is highest/lowest/leftest/rightest point in respect to anchor point.
    '''

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


def list_images(sort_order="asc"):
    images = os.listdir("images")

    if "processing" in images:
        images.remove("processing")

    # sort the files by date edited
    sort_prefix = 1
    if sort_order == "asc":
        sort_prefix = -1
    images.sort(key=lambda f: sort_prefix * os.path.getmtime(os.path.join("images", f)))
   
    return {
        "images": [i.split('.')[0] for i in images],
        "processing": [i.split('.')[0] for i in os.listdir("images/processing")]
    }


def read_cv2_image(id, path='images', readflags=cv2.IMREAD_COLOR):
    images = list(
        filter(lambda file: file.startswith(id), os.listdir(path)))
    if len(images) == 0:
        raise_error_response(f"image {id} does not exist", status=400)

    return cv2.imread(f"{path}/{images[0]}", readflags)


def list_cv2_images(readflags=cv2.IMREAD_COLOR):
    return [(read_cv2_image(id, readflags=readflags), id) for id in list_images()["images"]]
