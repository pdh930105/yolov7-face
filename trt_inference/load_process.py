# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, \
    resample_segments, clean_str
from utils.torch_utils import torch_distributed_zero_first
import pyrealsense2 as rs

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



class LoadRealSense:  # multiple IP or RTSP cameras
    def __init__(self, sources='rgb', img_size=640, stride=32, use_depth=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.sources = sources.lower()
        
        self.imgs = None
        self.depth_imgs = None
        
        assert self.sources=='rgb' or self.sources=='ir', f'Invalid source {self.sources}'
        fps = 30
        # Start thread to read frames from video stream
        if self.sources == 'rgb':        
            align = rs.align(rs.stream.color)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps) # opencv-aware format
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
            profile = pipeline.start(config)
            #intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
        else: # use ir stream
            align = rs.align(rs.stream.infrared)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, fps)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
            profile = pipeline.start(config)
            #intr = profile.get_stream(rs.stream.infrared).as_video_stream_profile().get_intrinsics()
        
        
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        frames = aligned_frames.get_color_frame() if self.sources == 'rgb' else aligned_frames.get_infrared_frame()
        depth_frames = aligned_frames.get_depth_frame()
        w, h = frames.get_width(), frames.get_height()
        self.fps = fps
        
        self.imgs = [np.asanyarray(frames.get_data())]

        thread = Thread(target=self.update, args=([pipeline, align]), daemon=True)
        print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
        thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, pipeline, align):
        # Read next stream frame in a daemon thread
        n = 0
        while pipeline.get_active_profile():
            n += 1
            # _, self.imgs[index] = cap.read()
            if n == 4:  # read every 4th frame
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
        
                frames = aligned_frames.get_color_frame() if self.sources == 'rgb' else aligned_frames.get_infrared_frame()
                depth_frames = aligned_frames.get_depth_frame()
                
                self.imgs = [np.asanyarray(frames.get_data())]
                self.depth_imgs = [np.asanyarray(depth_frames.get_data())]
                
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        depth_img0 = self.depth_imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]
        depth_img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in depth_img0]
        # Stack
        img = np.stack(img, 0)
        depth_img = np.stack(depth_img, 0)

        # Convert
        img = img.transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        depth_img = np.ascontiguousarray(depth_img)

        return self.sources, img, img0, depth_img, depth_img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
