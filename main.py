import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
N_OF_LANDMARKS = 81
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
        if(len(faces) == 0):
            raise "No face detected"
        # face's data
        self.face = faces[0]
        self.landmarks = predictor(image=self.gray, box=self.face)
        self.rows, self.cols, self.ch = img.shape
        self.points = [[self.landmarks.part(
            i).x, self.landmarks.part(i).y]for i in range(0, N_OF_LANDMARKS)]
        self.border = [[0, 0], [self.cols, 0],
                       [0, self.rows], [self.cols, self.rows]]

    # draws the image on plt
    def draw(self, img=None, include_points=True, window_name="Generic name", subplot=111):
        if img is None:
            img = self.img.copy()
        if include_points:
            cv2.rectangle(img=img, pt1=(self.face.left(), self.face.top()), pt2=(
                self.face.right(), self.face.bottom()), color=(0, 0, 255), thickness=4)
            for n in range(0, N_OF_LANDMARKS):
                x = self.landmarks.part(n).x
                y = self.landmarks.part(n).y

                # Draw a circle
                cv2.circle(img=img, center=(x, y), radius=3,
                           color=(0, 255, 0), thickness=-1)
        # [:,:,::-1] converts to RGB color space
        plt.subplot(subplot), plt.imshow(img[:, :, ::-1])
        plt.subplot(subplot), plt.imshow(img[:, :, ::-1])

    # applies from_img's face landmarks to image and draws it
    def draw_with_applied(self, from_img, anchor, window_name="Generic name", subplot=111):
        changes = [[from_img.points[i][j]-anchor.points[i][j] for j in range(2)]
                   for i in range(N_OF_LANDMARKS)]
        from_pts = from_img.points.copy()
        # coordinates of faces
        fx1, fy1 = from_img.face.left(), from_img.face.top()
        fx2, fy2 = from_img.face.right(), from_img.face.bottom()
        tx1, ty1 = self.face.left(), self.face.top()
        tx2, ty2 = self.face.right(), self.face.bottom()

        x_scale = (tx2-tx1)/(fx2-fx1)
        y_scale = (ty2-ty1)/(fy2-fy1)
        for i in range(N_OF_LANDMARKS):
            from_pts[i][0] = self.points[i][0] + x_scale*changes[i][0]  # x
            from_pts[i][1] = self.points[i][1] + y_scale*changes[i][1]  # y

        tform = PiecewiseAffineTransform()
        tform.estimate(np.float32(from_pts+self.border),  # border has to be self, so that image does not compress
                       np.float32(self.points + self.border))

        img = warp(self.img, tform, output_shape=(self.rows, self.cols))
        self.draw(img=img, include_points=False,
                  window_name=window_name, subplot=subplot)


if __name__ == "__main__":
    # read the image
    print("loading images")
    anchor = Image(cv2.imread("from_1.jpg"))
    from_img = Image(cv2.imread("from_2.jpg"))
    to_img = Image(cv2.imread("to_.jpg"))
    print("done loading")
    anchor.draw(window_name="anchor", subplot=234)
    from_img.draw(window_name="from", subplot=231)
    to_img.draw(window_name="to", subplot=236, include_points=False)
    to_img.draw(window_name="to", subplot=233)
    to_img.draw_with_applied(from_img, anchor,
                             window_name="mix", subplot=132)
    plt.show()
