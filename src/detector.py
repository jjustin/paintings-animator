from pydoc import describe
from unittest import skip
import cv2
import numpy as np
from storage.image import list_cv2_images

from helpers import Timer


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
    '''https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html'''
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

    def match(self, descriptors1, descriptors2, k=2):
        return self.matcher.knnMatch(descriptors1, k=2)

    def train(self, img):
        kp, des = self.detector.detectAndCompute(img, None)
        self.matcher.add([des])
        return kp, des


detector = _Detector()
'''detector is used singleton used for feature detection'''


class Template():
    '''
    Template is used to store a template image and its keypoints and descriptors
    '''

    def __init__(self, img):
        self.img = img
        self.keypoints, self.descriptors = detector.train(img)


templates = [Template(img)
             for img in list_cv2_images(readflags=cv2.IMREAD_GRAYSCALE)]
'''templates stores all images in storage to prevent loading them each time they are needed'''
detector.matcher.train()


def add_new_template(img_grayscale):
    '''
    add_new_template adds new template to templates list
    '''
    templates.append(Template(img_grayscale))
    detector.matcher.train()


RATIO_TRESHOLD = 0.7
''' https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html propeses 0.7'''


def detect(detecting_image, template_id):
    '''detect detects templates in the detecting_image and returns possible matches'''
    timer_all.start()
    detecting_image = cv2.cvtColor(detecting_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector(detecting_image, None)

    knn_matches = detector.match(descriptors, None, k=1)
    print(knn_matches[0])
    matchesMask = [[0, 0] for i in range(len(knn_matches))]
    good_matches = []
    for i, (m, n) in enumerate(knn_matches):
        if m.imgIdx != template_id:
            continue
        if m.distance < RATIO_TRESHOLD * n.distance:
            good_matches.append([m])
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    timer_all.end()
    template = templates[template_id]
    return cv2.drawMatchesKnn(detecting_image, keypoints, template.img, template.keypoints, knn_matches, None, **draw_params)

    return cv2.drawKeypoints(detecting_image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
