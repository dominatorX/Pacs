# Python plugin that supports loading batch of images in parallel
import cv2
import numpy
import threading
import os
import struct
from now.config import cfg
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait


_imread_executor_pool = ThreadPoolExecutor(max_workers=max(cfg.GLOBAL.BATCH_SIZE*2, 16))

class UnknownImageFormat(Exception):
    pass


def quick_imsize(file_path):
    """Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                height = int(h)
                width = int(w)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return height, width


def cv2_read_img_resize(path, read_storage, resize_storage, frame_size, grayscale):
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)
    resize_storage[:] = cv2.resize(read_storage, frame_size, interpolation=cv2.INTER_LINEAR)


def cv2_read_img(path, read_storage, grayscale):
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)


def wrf_read(path, wrf_storage):
    wrf_storage[:] = numpy.load("/export/data/wjc/learning/data/rain/wrf_all/"+path[-16:-8]+'.npy')


def quick_read_frames(path_list, im_h=None, im_w=None, resize=False, frame_size=None, channel=1):
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
        cv2_read_img(path=path_list[0], read_storage=read_storage[0, :, :], grayscale=True)
        if channel != 1:
            wrf_read(path_list[0], wrf_storage[0, :, :, :])
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i, :, :], True)
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
