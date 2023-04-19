from dof_trt_utils import YOLOEngine
import numpy as np
import cv2
import time
import os, sys
import argparse
from loguru import logger

class YOLOPredictor(YOLOEngine):
    def __init__(self, engine_path, logger, trt_nms, print_log):
        super(YOLOPredictor, self).__init__(engine_path, logger, trt_nms, print_log)
        self.n_classes = 1  # your model classes

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--engine", help="TRT engine Path")
	parser.add_argument("-i", "--image", help="image path")
	parser.add_argument("-o", "--output", default="output_trt.png",help="image output path")
	parser.add_argument("-l", "--log", default="./infer_trt.log",help="log path")
	parser.add_argument("-v", "--video",  help="video path or camera index ")
	parser.add_argument("--trt_nms", default=False, action="store_true",
					help="use trt_nms end2end engine (default : onnx_nms)")
	parser.add_argument("--print-log", default=False, action="store_true",
					help="use end2end engine")

	args = parser.parse_args()
	print(args)

	logger.add(args.log)

	pred = YOLOPredictor(engine_path=args.engine, logger=logger, trt_nms=args.trt_nms, print_log=args.print_log)
	img_path = args.image
	video = args.video
	if img_path:
		start_time = time.perf_counter()
		origin_img = pred.inference(img_path, conf=0.1)
	#print(origin_img.shape)
		if origin_img.shape[0] == 3:
			origin_img = origin_img.transpose(1, 2, 0)
		end_time = time.perf_counter()
		print(f"inference time : {end_time-start_time:.5f}s ({1/(end_time-start_time)} FPS)")
		cv2.imwrite("%s" %args.output, origin_img)
	"""
	if video:
		use_cam = True if video.isdigit() else False
		if args.v1:
			pred.detect_video(video, conf=0.5, use_cam=use_cam, end2end=args.end2end) # set 0 use a webcam
		else:
			pred.detect_video_v2(video, conf=0.5, use_cam=use_cam, end2end=args.end2end)
	if rs:
		pred.detect_rs(args.rs_type, conf=0.5)
	"""