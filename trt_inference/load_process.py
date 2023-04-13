# Dataset utils and dataloaders

import time
from itertools import repeat
from threading import Thread
import os, glob
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import pyrealsense2 as rs

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    print("input letterbox img shape :", img.shape)
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
    print("dw, dh : ", dw, dh)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    print("dw, dh result :", dw, dh)
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def rs_padding(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # 
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
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    img = cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=color)  # add border
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
        
        self.imgs = np.asanyarray(frames.get_data())
        self.depth_imgs = np.asanyarray(depth_frames.get_data())
        print("img shape :  ", self.imgs.shape)
        thread = Thread(target=self.update, args=([pipeline, align]), daemon=True)
        print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
        thread.start()
        print('')  # newline

        self.rect = 1 # realsense always load same frame shape
    def update(self, pipeline, align):
        # Read next stream frame in a daemon thread
        n = 0
        while pipeline.get_active_profile():
            n += 1
            # _, self.imgs[index] = cap.read()
            if n == 4:  # read every 4th frame
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                source_frames = aligned_frames.get_color_frame() if self.sources == 'rgb' else aligned_frames.get_infrared_frame()
                depth_frames = aligned_frames.get_depth_frame()
                print("depth frame : ", depth_frames)
                print("source frame : ", np.asanyarray(source_frames.get_data()).shape)
                self.imgs = np.asanyarray(source_frames.get_data())
                self.depth_imgs = np.asanyarray(depth_frames.get_data())
                
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
        print("pre image shape :",img0.shape, depth_img0.shape)
        img, ratio, _ = rs_padding(img0, self.img_size, auto=self.rect, stride=self.stride) 
        depth_img = rs_padding(depth_img0, self.img_size, auto=self.rect, stride=self.stride)[0]
        # Stack
        print("after img shape : ", img.shape, depth_img.shape)

        # Convert
        img = img.transpose(2,0,1)  # BGR to RGB, to 3x640x640
        img = img / 255.0
        img = np.ascontiguousarray(img, dtype=np.float32)
        depth_img = np.ascontiguousarray(depth_img, dtype=np.float16)

        return self.sources, img, img0, depth_img, depth_img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

class LoadImages:  # for inference
    def __init__(self, path, img_size=640 , stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            print("img0 shape :", img0.shape)
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            #print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        print("pre img shape :", img0.shape)
        img = letterbox(img0, self.img_size, stride=self.stride, auto=False)[0]
        print("after img shape :", img.shape)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 and normalize
        img = img / 255.0
        img = np.ascontiguousarray(img, dtype=np.float32)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
