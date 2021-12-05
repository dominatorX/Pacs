# Python plugin that supports loading batch of images in parallel
import cv2
import numpy
import threading
import os
import struct
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait


_imread_executor_pool = ThreadPoolExecutor(max_workers=16)


class UnknownImageFormat(Exception):
    pass


def cv2_read_img(path, read_storage):
    # read_storage[:] = numpy.load(path)
    read_storage[:] = cv2.imread(path, 0)


def wrf_read(path, wrf_storage):
    wrf_storage[:] = numpy.load("/export/data/wjc/learning/data/rain/wrf_all/"+path[-16:-8]+'.npy')


def quick_read_frames(path_list, im_h=None, im_w=None, channel=1):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    resize : bool, optional
    frame_size : None or tuple

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            print(path_list[i])
            raise IOError
    if channel == 1:
        read_storage = numpy.empty((img_num, 480, 480), dtype=numpy.uint8)
    else:
        read_storage = numpy.empty((img_num, 480, 480), dtype=numpy.float32)
        wrf_storage = numpy.empty((img_num, 162, 213, 4), dtype=numpy.float32)
    if img_num == 1:
        cv2_read_img(path=path_list[0], read_storage=read_storage[0, :, :])
        if channel != 1:
            wrf_read(path_list[0], wrf_storage[0, :, :, :])
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i, :, :])
            future_objs.append(obj)
        if channel != 1:
            future_objs = []
            for i in range(img_num):
                obj = _imread_executor_pool.submit(wrf_read, path_list[i], wrf_storage[i, :, :, :])
                future_objs.append(obj)
            wait(future_objs)
        wait(future_objs)
    if channel == 1:
        read_storage = read_storage[:, 170: 332, 102: 315].reshape((img_num, 1, im_h, im_w))
    else:
        read_storage = numpy.concatenate((read_storage[:, 170: 332, 102: 315,
                                          numpy.newaxis].reshape((img_num, im_h, im_w, 1)), wrf_storage),
                                         axis=3).transpose((0, 3, 1, 2))

    return read_storage[:, ::-1, ...]
