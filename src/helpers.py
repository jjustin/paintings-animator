from time import time_ns
import cv2 as cv
import json
from flask import abort, Response


def show(image, name="image"):
    cv.imshow(name, image)
    cv.waitKey(0)


def gpu_show(src: cv.cuda_GpuMat):
    img = src.download()
    show(img, name="gpu_image")


def cpu_to_gpu(image) -> cv.cuda_GpuMat:
    image_gpu = cv.cuda_GpuMat()
    image_gpu.upload(image)
    return image_gpu


def raise_error_response(error, status):
    resp = Response(json.dumps({"error": error}), status=status)
    abort(resp)


class Timer:
    times = {}
    times_count = {}

    def __init__(self, name) -> None:
        self._name = name
        if name not in Timer.times:
            Timer.times[name] = 0
            Timer.times_count[name] = 0
        self._passed = 0
        self._running = False

    def start(self, *, phony=False):
        if phony:
            return self
        self._running = True
        self._start_time = time_ns()
        return self

    def stop(self, *, phony=False):
        if phony:
            return self
        if self._running:
            self._passed += time_ns() - self._start_time
        else:
            print("Timer stopped when not running")
        self._running = True

    def end(self, *, phony=False):
        if phony:
            return self
        self.stop()
        Timer.times[self._name] += self._passed
        self._passed = 0
        Timer.times_count[self._name] += 1

    @staticmethod
    def avg(name):
        avg = Timer.times[name] / Timer.times_count[name]
        return f"{to_ms(avg)}ms"

    @staticmethod
    def count(name):
        out = Timer.times_count[name]
        return f"{out}"

    @staticmethod
    def total(name):
        out = Timer.times[name]
        return f"{to_ms(out)}ms"

    @staticmethod
    def stat():
        return {
            name: {
                "perLoop": Timer.avg(name),
                "loops": Timer.count(name),
                "totalTime": Timer.total(name),
            }
            for name in Timer.times
            if Timer.times_count[name] != 0
        }


def to_ms(ns_time):
    return ns_time/10**6
