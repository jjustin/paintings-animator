from time import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from storage.image import list_cv2_images

from helpers import Timer

Point = Tuple[int, int]
Corners = List[Point]

timer_all = Timer("detect")
timer_keypoints = Timer("detect_keypoints")


class _Detector():
    '''
    _Decetor is wrapper interface to cv2's detector interfaces
    '''
    # check if cuda implementation exists if this turns out to be the bottleneck

    d_func = cv2.SIFT_create
    # d_func = cv2.FastFeatureDetector_create
    # d_func = cv2.ORB_create
    d_args = {}

    # matcher = cv2.BFMatcher()

    FLANN_INDEX_KDTREE = 1
    '''see https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html'''

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    def __init__(self):
        self.detector = self.d_func(**self.d_args)
        self.matcher = cv2.FlannBasedMatcher(
            self.index_params, self.search_params)

    def __call__(self, img, mask, skip_timer=False):
        timer_keypoints.start(phony=skip_timer)
        keypoints, descriptors = self.detector.detectAndCompute(img, mask)
        timer_keypoints.end(phony=skip_timer)
        return keypoints, descriptors

    def detect(self, img, mask):
        return self.detector.detect(img, mask)

    def match(self, descriptors, k=2):
        return self.matcher.knnMatch(descriptors, k=k)

    def add(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        self.matcher.add([des])
        return kp, des


detector = _Detector()
'''detector is singleton used for feature detection'''


class Template():
    '''
    Template is used to store a template image and its keypoints and descriptors.
    Template is autmatically added to the detector when created.
    '''

    def __init__(self, img, img_id: str):
        self.img = img
        self.img_id = img_id
        self.keypoints, self.descriptors = detector.add(img)


templates = [Template(img, img_id)
             for (img, img_id) in list_cv2_images(readflags=cv2.IMREAD_GRAYSCALE)]
'''templates stores all images in storage to prevent loading them each time they are needed'''


def add_new_template(img_grayscale, img_id):
    '''
    add_new_template adds new template to templates list
    '''
    print("registering template for " + img_id)
    templates.append(Template(img_grayscale, img_id))


RATIO_TRESHOLD = 0.7
''' https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html propeses 0.7'''


def detect(detecting_image, skip_images=[], good_match_threshold=10):
    '''
    detect detects templates in the detecting_image and returns possible matches
    
    good_match_threshold - number of keypoints that must match to be considered a match
    skip_images - list of image ids to skip
    '''
    timer_all.start()

    # Scale the image to height of max 1440 to reduce processing time
    print("input size: ", detecting_image.shape)

    if detecting_image.shape[0] > 1080:
        scale_factor = 1080 / detecting_image.shape[0]

        detecting_image = cv2.resize(
            detecting_image, (0, 0), fx=scale_factor, fy=scale_factor)
        print("input size after resize: ", detecting_image.shape)

    detecting_image = cv2.cvtColor(detecting_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector(detecting_image, None)

    knn_matches = detector.match(descriptors, k=2)

    matches_counter: Dict[str, List] = {}
    for i, (m, n) in enumerate(knn_matches):
        if m.distance < RATIO_TRESHOLD * n.distance:
            if m.imgIdx not in matches_counter:
                print(m.imgIdx)
                matches_counter[m.imgIdx] = []
            matches_counter[m.imgIdx].append(m)

    out = []

    detecting_h, detecting_w = detecting_image.shape

    for i in matches_counter.keys():
        good_matches = matches_counter[i]
        template: Template = templates[i]
        print(f"{template.img_id} contains {len(good_matches)} keypoint matches")
        if template.img_id in skip_images:
            continue

        # skip images with not enough matched keypoints
        if len(good_matches) < good_match_threshold:
            continue

        # create arrays of points in gallery and on image
        template_points = np.empty((len(good_matches), 2), dtype=np.float32)
        gallery = np.empty((len(good_matches), 2), dtype=np.float32)

        for i, m in enumerate(good_matches):
            # TODO: shorter?
            template_points[i, 0] = template.keypoints[m.trainIdx].pt[0]
            template_points[i, 1] = template.keypoints[m.trainIdx].pt[1]
            gallery[i, 0] = keypoints[m.queryIdx].pt[0]
            gallery[i, 1] = keypoints[m.queryIdx].pt[1]

        H, _ = cv2.findHomography(template_points, gallery, cv2.RANSAC)

        # transform template corner points to gallery points
        h, w = template.img.shape
        template_corners = np.array([
            [[0, 0]],
            [[w, 0]],
            [[0, h]],
            [[w, h]],
        ], dtype=np.float32)

        painting_corners = cv2.perspectiveTransform(template_corners, H)
        corners = [[point[0]/detecting_w, point[1]/detecting_h]
                   for [point] in painting_corners]

        out.append({
            "name": template.img_id,
            "corners": corners,
        })

        # draw_params = dict(matchColor=(0, 255, 0),
        #                    singlePointColor=(255, 0, 0),
        #                    matchesMask=matchesMask,
        #                    flags=cv2.DrawMatchesFlags_DEFAULT)

        # kp_img = cv2.drawMatchesKnn(detecting_image, keypoints, template.img,
        #                             template.keypoints, knn_matches, None, **draw_params)
        cv2.polylines(detecting_image, [np.int32(painting_corners)],
                      True, (0, 255, 0), 1)

    # For debug purposes
    cv2.imwrite(f"debug/detected_{time()}.png", detecting_image)

    timer_all.end()

    return out
