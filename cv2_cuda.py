# Custom implementation of piecewise affine transformation

import json
from multiprocessing import Pool
from time import time_ns
import numpy as np
import cv2 as cv

from helpers import cpu_to_gpu, show, Timer

has_cuda = cv.cuda.getCudaEnabledDeviceCount() > 0
if not has_cuda:
    print("WARN: Not using cuda for image warping")


class Point(object):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

    def __hash__(self) -> int:
        return hash(self.x) + hash(self.y)

    def aslist(self):
        return [self.x, self.y]


class Transformer():
    def __init__(self, img, src, dst) -> None:
        self.img = img
        if has_cuda:
            self.src = src
            self.dst = dst
            return
        self.dst_img = None
        self.mapping = self._get_point_mapping(src, dst)
        w, h = self._img_size()
        # w+1, h+1 to prevent out of bounds for border points
        self.subdiv = cv.Subdiv2D((0, 0, w+1, h+1))
        self.subdiv.insert(src)

    @staticmethod
    def _get_point_mapping(src, dst):
        '''returns mapping from src to dst'''
        src = Transformer._points_as_objects(src)
        return dict(zip(src, dst))

    def warp_affine_pw(self):
        '''preforms affine piecewise transformation and returns transformed image in CPU memory'''
        if has_cuda:
            return cv.cuda.warpPiecewiseAffine(self.img, self.src, self.dst).download()
            
        w, h = self._img_size()
        self.dst_img = np.ones((h, w, 3), np.uint8)

        tris = self.subdiv.getTriangleList()
        tris = np.resize(tris, (tris.shape[0], 3, 2))
        a = np.array(list(map(self._get_mapping_pairs, tris)))

        for src, dst in a:
            t = Timer("warp").start()
            transformed = self.warp_affine(src, dst)
            t.end()
            t = Timer("join").start()
            self._join_on_dst(transformed, dst)
            t.end()

        img = self.img.download()
        img_out = np.where(self.dst_img != 1, self.dst_img, img)
        return img_out

    def warp_affine(self, src, dst):
        '''warp_affine performs single affine transformation of image and returns image on CPU'''
        affine = cv.getAffineTransform(src, dst)
        if has_cuda:
            warp = cv.cuda.warpAffine
        else:
            warp = cv.warpAffine

        res = warp(self.img, affine, self._img_size(), borderValue=())

        if has_cuda:
            return res.download()
        return res

    def _join_on_dst(self, transformed, poly):
        '''_join_on_dst moves transormed pixels inside poly to destination image'''
        # get max/min points, where transformed can be cropped
        t = Timer("min finding in _join_on_dst").start()
        x = poly[:, 0]
        y = poly[:, 1]
        minx, miny = round(np.min(x)), round(np.min(y))
        maxx, maxy = round(np.max(x)), round(np.max(y))
        if minx == maxx or miny == maxy:
            return

        poly_cropped = poly - [minx, miny]
        t.end()

        t = Timer("joining in _join_on_dst").start()
        # crop the image to reduce size of masks, eventually reducing computing time
        transformed = transformed[miny:maxy, minx:maxx]

        # part of destination image where transformed will be applied
        dst_img_overlap = self.dst_img[miny:maxy, minx:maxx]
        # mask pixels outside of poly
        mask = np.zeros_like(transformed)
        cv.fillPoly(mask, [poly_cropped.astype(np.int32)], (1, 1, 1))
        # prevent overlaping of copied data
        mask[dst_img_overlap != 1] = 0
        # apply mask to transformed leaving only pixels inside poly
        transformed = cv.multiply(transformed, mask)

        # set pixels outside of poly to 1. This way dst_image will have unset pixels set to 1 after multiplication
        mask = 1-mask
        transformed += mask

        self.dst_img[miny:maxy, minx:maxx] = cv.multiply(
            dst_img_overlap, transformed)
        t.end()

    @staticmethod
    def _point_as_object(pt):
        """returns single point as Point object"""
        return Point(pt[0], pt[1])

    @staticmethod
    def _points_as_objects(pts):
        '''
        param pts: (n,2) ndarray of points to map to
        '''
        return np.array(list(map(Transformer._point_as_object, pts)), dtype=object)

    def _get_mapping_pairs(self, src):
        '''
        param src: list of origin points to find the destination points for
        return: nparray ["source points", "corresponding destination points"]
        '''
        src_pt = Transformer._points_as_objects(src)
        return np.array([
            src,
            [self.mapping[p] for p in src_pt]
        ], dtype=np.float32)

    def _img_size(self):
        if has_cuda:
            return self.img.size()
        h, w = self.img.shape[:2]
        return w, h


if __name__ == "__main__":
    totalTimer = Timer("total").start()

    def pw_af_transform(img, src, dst):
        if has_cuda:
            img = cpu_to_gpu(img)
        t = Transformer(img, src, dst)
        return t.warp_affine_pw()

    def to_s(ns_time):
        return ns_time/1000000000

    t = Timer("img load").start()
    image_cpu = cv.imread('image7.jpg')
    # imshow(image_cpu)

    image_gpu = cpu_to_gpu(image_cpu)
    # gpu_imshow(image_gpu)
    t.end()

    # pts_fr = [[105, 136], [107, 146], [110, 155], [113, 165], [118, 172], [125, 179], [134, 184], [144, 187], [153, 187], [161, 184], [167, 179], [172, 172], [176, 164], [177, 155], [177, 146], [177, 137], [177, 127], [113, 129], [118, 124], [124, 122], [131, 123], [139, 125], [149, 123], [155, 119], [161, 117], [168, 117], [172, 121], [146, 132], [148, 138], [149, 144], [151, 151], [143, 156], [147, 156], [151, 157], [154, 156], [155, 154], [
    #     121, 135], [125, 133], [130, 132], [134, 134], [130, 136], [125, 137], [154, 132], [157, 128], [162, 127], [166, 128], [163, 131], [158, 132], [137, 167], [143, 166], [148, 165], [151, 165], [154, 163], [157, 163], [161, 164], [159, 168], [155, 169], [152, 170], [149, 170], [144, 169], [140, 167], [148, 166], [151, 166], [154, 165], [161, 164], [155, 165], [152, 166], [148, 166], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153]]
    # pts_to = [[105, 137], [107, 147], [110, 156], [113, 165], [118, 173], [125, 180], [134, 185], [144, 188], [153, 188], [161, 185], [167, 180], [172, 173], [176, 165], [177, 156], [178, 147], [178, 138], [177, 128], [113, 129], [118, 124], [124, 122], [131, 122], [138, 124], [151, 123], [157, 119], [163, 117], [169, 117], [173, 121], [146, 132], [148, 139], [149, 145], [151, 152], [143, 157], [147, 157], [151, 158], [154, 157], [156, 155], [121, 136], [125, 133], [130, 133], [
    #     135, 135], [130, 137], [125, 138], [155, 133], [158, 129], [163, 128], [167, 129], [164, 132], [159, 133], [137, 168], [143, 166], [148, 165], [151, 165], [154, 164], [158, 164], [162, 165], [159, 169], [155, 170], [152, 171], [149, 171], [144, 170], [140, 168], [148, 167], [151, 167], [154, 166], [161, 165], [155, 166], [152, 167], [148, 167], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153]]
    pts_fr = [[105, 137], [107, 147], [110, 156], [113, 165], [118, 173], [125, 180], [134, 185], [144, 188], [153, 188], [161, 185], [167, 180], [172, 173], [176, 165], [177, 156], [178, 147], [178, 138], [177, 128], [113, 129], [118, 124], [124, 122], [131, 122], [138, 124], [151, 123], [157, 119], [163, 117], [169, 117], [173, 121], [146, 132], [148, 139], [149, 145], [151, 152], [143, 157], [147, 157], [151, 158], [154, 157], [156, 155], [121, 136], [125, 133], [130, 133], [
        135, 135], [130, 137], [125, 138], [155, 133], [158, 129], [163, 128], [167, 129], [164, 132], [159, 133], [137, 168], [143, 166], [148, 165], [151, 165], [154, 164], [158, 164], [162, 165], [159, 169], [155, 170], [152, 171], [149, 171], [144, 170], [140, 168], [148, 167], [151, 167], [154, 166], [161, 165], [155, 166], [152, 167], [148, 167], [102, 115], [179, 115], [102, 191], [179, 191], [140, 115], [140, 191], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]]
    pts_to = [[105.0, 137.23225806451612], [107.0, 147.23225806451612], [110.0, 156.23225806451612], [113.0, 165.23225806451612], [118.0, 173.0], [125.0, 180.0], [134.0, 185.0], [144.0, 188.0], [153.0, 187.76774193548388], [161.0, 185.0], [167.0, 180.0], [172.0, 173.0], [176.0, 164.76774193548388], [177.0, 156.0], [178.0, 147.0], [178.0, 138.0], [177.0, 128.0], [113.0, 129.0], [118.0, 124.0], [124.0, 122.0], [131.0, 122.0], [138.0, 124.0], [151.0, 123.0], [157.0, 119.0], [163.0, 117.0], [169.0, 117.0], [173.0, 121.0], [146.0, 132.0], [148.0, 139.0], [149.0, 145.0], [151.0, 152.0], [143.0, 157.0], [147.0, 157.0], [151.0, 158.0], [154.0, 157.0], [156.0, 155.0], [
        121.0, 136.0], [125.0, 133.0], [130.0, 133.0], [135.0, 135.0], [130.0, 137.0], [125.0, 138.0], [155.0, 133.0], [158.0, 129.0], [163.0, 128.0], [167.0, 129.0], [164.0, 132.0], [159.0, 133.0], [136.76699029126215, 168.0], [143.0, 166.0], [148.0, 165.0], [151.0, 165.0], [154.0, 164.0], [158.0, 164.0], [162.0, 165.0], [159.0, 169.0], [155.0, 170.0], [152.0, 171.0], [149.0, 171.0], [144.0, 170.0], [140.0, 168.0], [148.0, 167.0], [151.0, 167.0], [154.0, 166.0], [161.0, 165.0], [155.0, 166.0], [152.0, 167.0], [148.0, 167.0], [102, 115], [179, 115], [102, 191], [179, 191], [140, 115], [140, 191], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]]

    loopTimer = Timer("main loop").start()
    for i in range(24):
        t = Timer("pw_af_transform").start()
        pw_af_transform(image_cpu, pts_fr, pts_to)
        t.end()
    loopTimer.end()
    totalTimer.end()

    print("cuda:", has_cuda)
    print(json.dumps(Timer.stat()))
