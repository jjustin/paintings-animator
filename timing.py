# Experimental comparison of multiple ways of doing the same thing
# Might come in handy when writing the report

from multiprocessing import Pool
from time import time_ns
import numpy as np
from scipy import spatial
import cv2 as cv

from cv2_cuda import Img
####
# Dict compose
####

# class Point:
#     def __init__(self, x, y) -> None:
#         self.x = x
#         self.y = y

#     def __eq__(self, __o: object) -> bool:
#         return self.x == __o.x and self.y == __o.y

#     def __hash__(self) -> int:
#         return hash(self.x) + hash(self.y)


# def asObj(pt):
#     return Point(pt[0], pt[1])


# pts_fr = [[105, 136], [107, 146], [110, 155], [113, 165], [118, 172], [125, 179], [134, 184], [144, 187], [153, 187], [161, 184], [167, 179], [172, 172], [176, 164], [177, 155], [177, 146], [177, 137], [177, 127], [113, 129], [118, 124], [124, 122], [131, 123], [139, 125], [149, 123], [155, 119], [161, 117], [168, 117], [172, 121], [146, 132], [148, 138], [149, 144], [151, 151], [143, 156], [147, 156], [151, 157], [154, 156], [155, 154], [
#     121, 135], [125, 133], [130, 132], [134, 134], [130, 136], [125, 137], [154, 132], [157, 128], [162, 127], [166, 128], [163, 131], [158, 132], [137, 167], [143, 166], [148, 165], [151, 165], [154, 163], [157, 163], [161, 164], [159, 168], [155, 169], [152, 170], [149, 170], [144, 169], [140, 167], [148, 166], [151, 166], [154, 165], [161, 164], [155, 165], [152, 166], [148, 166], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]]
# pts_to = [[105, 137], [107, 147], [110, 156], [113, 165], [118, 173], [125, 180], [134, 185], [144, 188], [153, 188], [161, 185], [167, 180], [172, 173], [176, 165], [177, 156], [178, 147], [178, 138], [177, 128], [113, 129], [118, 124], [124, 122], [131, 122], [138, 124], [151, 123], [157, 119], [163, 117], [169, 117], [173, 121], [146, 132], [148, 139], [149, 145], [151, 152], [143, 157], [147, 157], [151, 158], [154, 157], [156, 155], [121, 136], [125, 133], [130, 133], [
#     135, 135], [130, 137], [125, 138], [155, 133], [158, 129], [163, 128], [167, 129], [164, 132], [159, 133], [137, 168], [143, 166], [148, 165], [151, 165], [154, 164], [158, 164], [162, 165], [159, 169], [155, 170], [152, 171], [149, 171], [144, 170], [140, 168], [148, 167], [151, 167], [154, 166], [161, 165], [155, 166], [152, 167], [148, 167], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]]

# start_time = time_ns()
# for i in range(1000):
#     d = {}
#     for p in range(len(fr)):
#         d[fr[p]] = to[p]

# print(time_ns() - start_time)

# start_time = time_ns()
# for i in range(1000):
#     d = dict(zip(fr, to))

# print(time_ns() - start_time)

#####
# Two lists
#####


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


# x = np.array([[105, 136], [107, 146], [110, 155], [113, 165], [118, 172], [125, 179], [134, 184], [144, 187], [153, 187], [161, 184], [167, 179], [172, 172], [176, 164], [177, 155], [177, 146], [177, 137], [177, 127], [113, 129], [118, 124], [124, 122], [131, 123], [139, 125], [149, 123], [155, 119], [161, 117], [168, 117], [172, 121], [146, 132], [148, 138], [149, 144], [151, 151], [143, 156], [147, 156], [151, 157], [154, 156], [155, 154], [
#     121, 135], [125, 133], [130, 132], [134, 134], [130, 136], [125, 137], [154, 132], [157, 128], [162, 127], [166, 128], [163, 131], [158, 132], [137, 167], [143, 166], [148, 165], [151, 165], [154, 163], [157, 163], [161, 164], [159, 168], [155, 169], [152, 170], [149, 170], [144, 169], [140, 167], [148, 166], [151, 166], [154, 165], [161, 164], [155, 165], [152, 166], [148, 166], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]])
# y = np.array([[105, 136],  [113, 165], [125, 137]])
# z = np.array([[105, 137], [107, 147], [110, 156], [113, 165], [118, 173], [125, 180], [134, 185], [144, 188], [153, 188], [161, 185], [167, 180], [172, 173], [176, 165], [177, 156], [178, 147], [178, 138], [177, 128], [113, 129], [118, 124], [124, 122], [131, 122], [138, 124], [151, 123], [157, 119], [163, 117], [169, 117], [173, 121], [146, 132], [148, 139], [149, 145], [151, 152], [143, 157], [147, 157], [151, 158], [154, 157], [156, 155], [121, 136], [125, 133], [130, 133], [
#     135, 135], [130, 137], [125, 138], [155, 133], [158, 129], [163, 128], [167, 129], [164, 132], [159, 133], [137, 168], [143, 166], [148, 165], [151, 165], [154, 164], [158, 164], [162, 165], [159, 169], [155, 170], [152, 171], [149, 171], [144, 170], [140, 168], [148, 167], [151, 167], [154, 166], [161, 165], [155, 166], [152, 167], [148, 167], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]])

# st = time_ns()
# for _ in range(1000):
#     a = np.array([i for i, _x in enumerate(x)
#                   for _y in y if _x[0] == _y[0] and _x[1] == _y[1]], dtype=x.dtype)
#     a = z[a]
# print(time_ns() - st)


# def f(big_array, small_array, tolerance=0.5):
#     tree_big = spatial.cKDTree(big_array)
#     tree_small = spatial.cKDTree(small_array)
#     return tree_small.query_ball_tree(tree_big, r=tolerance)


# st = time_ns()
# for _ in range(1000):
#     ## Not complete
#     a = f(x, y)
# print(time_ns() - st)


# def as_str(pt):
#     return "{}_{}".format(pt[0], pt[1])


# st = time_ns()
# for _ in range(1000):
#     d = dict(zip(map(as_str, x), z))
#     a = np.array([d[as_str(p)] for p in y])
# print(time_ns() - st)


# x = np.array(list(map(asObj, x)), dtype=object)
# y = np.array(list(map(asObj, y)), dtype=object)
# # z = np.array(list(map(asObj, z)), dtype=object)


# st = time_ns()
# for _ in range(1000):
#     a = np.array([i for i, _x in enumerate(x)
#                   for _y in y if _x == _y])
#     a = z[a]
# print(time_ns() - st)

# st = time_ns()
# for _ in range(1000):
#     a = z[np.isin(x, y)]
# print(time_ns() - st)


# st = time_ns()
# for _ in range(1000):
#     dict_map = dict(zip(x, z))
#     a = np.array([dict_map[pt] for pt in y])
# print(time_ns() -st)


# print(a)

def asObj(pt):
    return Point(pt[0], pt[1])


x = np.array([[105, 136], [107, 146], [110, 155], [113, 165], [118, 172], [125, 179], [134, 184], [144, 187], [153, 187], [161, 184], [167, 179], [172, 172], [176, 164], [177, 155], [177, 146], [177, 137], [177, 127], [113, 129], [118, 124], [124, 122], [131, 123], [139, 125], [149, 123], [155, 119], [161, 117], [168, 117], [172, 121], [146, 132], [148, 138], [149, 144], [151, 151], [143, 156], [147, 156], [151, 157], [154, 156], [155, 154], [
    121, 135], [125, 133], [130, 132], [134, 134], [130, 136], [125, 137], [154, 132], [157, 128], [162, 127], [166, 128], [163, 131], [158, 132], [137, 167], [143, 166], [148, 165], [151, 165], [154, 163], [157, 163], [161, 164], [159, 168], [155, 169], [152, 170], [149, 170], [144, 169], [140, 167], [148, 166], [151, 166], [154, 165], [161, 164], [155, 165], [152, 166], [148, 166], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]])
z = np.array([[105, 137], [107, 147], [110, 156], [113, 165], [118, 173], [125, 180], [134, 185], [144, 188], [153, 188], [161, 185], [167, 180], [172, 173], [176, 165], [177, 156], [178, 147], [178, 138], [177, 128], [113, 129], [118, 124], [124, 122], [131, 122], [138, 124], [151, 123], [157, 119], [163, 117], [169, 117], [173, 121], [146, 132], [148, 139], [149, 145], [151, 152], [143, 157], [147, 157], [151, 158], [154, 157], [156, 155], [121, 136], [125, 133], [130, 133], [
    135, 135], [130, 137], [125, 138], [155, 133], [158, 129], [163, 128], [167, 129], [164, 132], [159, 133], [137, 168], [143, 166], [148, 165], [151, 165], [154, 164], [158, 164], [162, 165], [159, 169], [155, 170], [152, 171], [149, 171], [144, 170], [140, 168], [148, 167], [151, 167], [154, 166], [161, 165], [155, 166], [152, 167], [148, 167], [102, 116], [179, 116], [102, 190], [179, 190], [140, 116], [140, 190], [179, 153], [102, 153], [0, 0], [320, 0], [0, 400], [320, 400]])
x_pts = np.array(list(map(asObj, x)), dtype=object)

subdiv = cv.Subdiv2D((0, 0, 321, 401))
subdiv.insert(x.tolist())

img_cpu = cv.imread('image7.jpg')
img = cv.cuda_GpuMat(img_cpu.size)
img.upload(img_cpu)


def warp_affine(e):
    src, dst = e
    affine = cv.getAffineTransform(src, dst)
    return cv.cuda.warpAffine(img, affine, (320, 400))


def warp_affine_cpu(e):
    src, dst = e
    affine = cv.getAffineTransform(src, dst)
    return cv.warpAffine(img_cpu, affine, (320, 400))


# transform lists of flattened points to list of lists of points
def to_points(tri: np.ndarray):
    return np.resize(tri, (tri.shape[0], 3, 2))


dict_map = dict(zip(x_pts, z))


# get pairs of tripoints, first element are src edges, second elemets are dst edges
def get_pairs(src):
    src_pt = np.array(list(map(asObj, src)), dtype=object)
    return np.array([src, [dict_map[p] for p in src_pt]], dtype=np.float32)


st = time_ns()


pairs = to_points(subdiv.getTriangleList())
a = np.array(list(map(get_pairs, pairs)))
print(time_ns()-st)

# # for loop
# st = time_ns()
# for i in range(30):
#     for e in a:
#         warp_affine(e)
# print(time_ns()-st)

# # map
# st = time_ns()
# for i in range(30):
#     x = list(map(warp_affine, a))
# print(time_ns()-st)


# # map cpu
# st = time_ns()
# x = list(map(warp_affine_cpu, a))
# print(time_ns()-st)

# # pool map - errors occur
# pool = Pool()
# st = time_ns()
# x = pool.map_async(f, a).wait()
# print(time_ns()-st)

#####
# crops
#####

st = time_ns()
for i in range(12000):
    im = cv.cuda_GpuMat(img, (30, 0, 290, 200))
print(time_ns()-st)

st = time_ns()
for i in range(12000):
    im = img_cpu[0:200, 30:320]
print(time_ns()-st)

st = time_ns()
for i in range(12000):
    im = img.download()
    im = im[0:200, 30:320]
print(time_ns()-st)
