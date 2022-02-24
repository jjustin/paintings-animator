# Comparison of cv2's cuda and cpu affine transformation

from time import time_ns
import numpy as np
import cv2 as cv


def imshow(image, name="image"):
    cv.imshow(name, image)
    cv.waitKey(0)


def gpu_imshow(src: cv.cuda_GpuMat):
    img = src.download()
    imshow(img, name="gpu_image")


def cpu_to_gpu(image) -> cv.cuda_GpuMat:
    image_gpu = cv.cuda_GpuMat
    image_gpu.upload(image)


image_cpu = cv.imread('image7.jpg')
# imshow(image_cpu)
image_gpu = cv.cuda_GpuMat()
image_gpu.upload(image_cpu)
# gpu_imshow(image_gpu)

time = {
    "gpu": 0,
    "cpu": 0,
}


pts_fr = np.float32([[0, 0], [100, 0], [100, 100]])
pts_to = np.float32([[0, 100], [100, 0], [0, 0]])

rows, cols = image_cpu.shape[:2]

start_time = time_ns()
for i in range(0, 12000):
    if i % 100 == 0:
        print(i)

    affine = cv.getAffineTransform(pts_fr, pts_to)
    im = cv.cuda.warpAffine(image_gpu, affine, (cols, rows))

    # if i == 0:
    #     gpu_imshow(im)
time["gpu"] = + time_ns() - start_time

start_time = time_ns()
for i in range(0, 12000):
    if i % 100 == 0:
        print(i)

    affine = cv.getAffineTransform(pts_fr, pts_to)
    im = cv.warpAffine(image_cpu, affine, (cols, rows))
time["cpu"] = + time_ns() - start_time

print(time)
