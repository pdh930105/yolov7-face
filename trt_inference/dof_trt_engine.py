import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import time
import pyrealsense2 as rs
import sys
from load_process import LoadRealSense, LoadImages
from dof_trt_utils import compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d, vis_end2end, vis, preproc, preproc_pad

class YOLOEngine(object):
    def __init__(self, engine_path, logger, model_name='yolo',trt_nms=True,print_log=False):
        self.engine_name= model_name
        self.mean = None
        self.std = None
        self.n_classes = 1
        self.nkpt = 5
        self.use_onnx_trt = False
        self.class_names = ['face']
        self.logger =logger
        self.print_log = print_log
        self.trt_nms = trt_nms

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        self.logger.info(f"starting YOLO engine {self.engine_name}")
        
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.logger.info(f"\n output size : {size} \n dtype : {dtype} \n input_size : {self.imgsz} \n ")
        self.logger.info(f"Finish load YOLO engine {engine_path}")

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        # 여기에 nms 추가할 수는 없나
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        #self.logger.info("finished inference")
        data = [out['host'] for out in self.outputs]
        return data

    def forward_yolo(self, img):
        """
        forward_yolo run yolo inference (using onnx_nms version only)
        input : img after preprocessing (np.array)
        output : 1 detect_box information and keypoint (largest box size) 
        """
        yolo_output, nms_result = self.infer(img)
        yolo_output = yolo_output.reshape(-1, 6 + self.nkpt*3) # (xywh + box + cls (6)) + self.nkpt * 3
        nms_idx = np.unique(nms_result)
        if len(nms_idx) > 0: # exists detect box
            detect_box = yolo_output[nms_idx[1:]] # remove zero index 
            if len(detect_box) > 1: # get multiple box
                detect_box = yolo_output[np.argmax((yolo_output[nms_idx, 2] * yolo_output[nms_idx, 3]))] # select max box size
     
            return detect_box.flatten(), True
        else:
            return None, False


    def inference(self, img_path, conf=0.5):
        start_time = time.perf_counter()
        origin_img = cv2.imread(img_path)
        #img, ratio = preproc_pad(origin_img, self.imgsz, self.mean, self.std)
        #data = self.infer(img)
        
        if self.trt_nms:
            prepare_img_time = time.perf_counter()
            img, ratio = preproc_pad(origin_img, self.imgsz, self.mean, self.std)
            preprocess_img_time = time.perf_counter()
            data = self.infer(img)
            infer_time = time.perf_counter()
            num, final_boxes, final_scores, final_cls_inds = data

            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

            if dets is not None:
                final_boxes, final_scores, final_classes = dets[:,:4], dets[:, 4], dets[:, 5]
                origin_img = vis_end2end(origin_img, final_boxes, final_scores, final_classes,
                                conf=conf, class_names=self.class_names, box_type='xyxy')
                post_process_time = time.perf_counter()
                if self.print_log:
                    for trt_output in data:
                        print(trt_output.shape)
                    print("num : ", num)
                    print("final boxes: ", final_boxes)
                    print("score : ", final_scores[:num[0]])
                    print("final cls inds :", final_cls_inds)
                    print(f"total inference time : {post_process_time-start_time:.3f}s ({1/(post_process_time-start_time):.3f} FPS)")
                    print(f"img read time : {prepare_img_time-start_time:.3f}s")
                    print(f"img preprocess time : {preprocess_img_time-prepare_img_time:.3f}s")
                    print(f"infer time : {infer_time-preprocess_img_time:.3f}s")
                    print(f"post process time : {post_process_time - infer_time:.3f}")

            return origin_img, final_boxes, None
        else: # using onnx nms
            resized_img, resized_img_tran = preproc(origin_img, self.imgsz)
            yolo_output, nms_result = self.infer(resized_img_tran)
            yolo_output = yolo_output.reshape(-1, 6 + self.nkpt*3) # (xywh + box + cls (6)) + self.nkpt * 3
            nms_idx = np.unique(nms_result)
            if len(nms_idx) > 0: # exists detect box
                detect_box = yolo_output[nms_idx[1:]] # remove zero index 
                #if len(detect_box) > 0:
                #    detect_box = yolo_output[np.argmax((yolo_output[nms_idx, 2] * yolo_output[nms_idx, 3]))]
                origin_img = vis_end2end(resized_img, detect_box[:, :4], detect_box[:, 5], np.ones_like(detect_box[:, 5]), conf=conf, class_names=self.class_names, box_type='xywh')

        return origin_img


class DoFEngine(object):
    def __init__(self, engine_path, model_name, logger, print_log=False):
        self.model_name = model_name
        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,-1) # imagenet mean
        self.img_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,-1) # imagenet std
        self.logger =logger
        self.print_log = print_log

        self.logger.info(f"Starting DoF engine : {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        self.logger.info(f"\n output size : {size} \n dtype : {dtype} \n input_size : {self.imgsz} \n ")
        self.logger.info(f"Finish load DoF engine {engine_path}")

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        # 여기에 nms 추가할 수는 없나
        
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        #self.logger.info("finished inference")
        data = [out['host'] for out in self.outputs]
        return data

    def dof_forward(self, img):
        preproc_img = self.preprocess(img)
        output_list = self.infer(preproc_img)
        predictions = output_list[0].reshape(1, -1)
        print("dof prediction : ", predictions)
        pitch, yaw, roll = self.postprocess(predictions)
        return pitch, yaw, roll
    
    def preprocess(self, img):
        """
        preprocess: resized image and normalize (standardization) 
        input : cropped face img (h,w,c)
        output : resize and normalize image
        """
        if img.shape[0]==3:
            resized_img = img.transpose(1,2,0) # (3, h, w) to (h, w, 3)
        else:
            resized_img = img.copy()
        resized_img = cv2.resize(resized_img, dsize=self.imgsz, interpolation=cv2.INTER_AREA)
        resized_img = (resized_img - self.img_mean) / self.img_std
        resized_img = np.ascontiguousarray(resized_img.transpose(2,0,1), np.float32) # (self.imgsz, self.imgsz, 3) to (3, self.imgsz, self.imgsz)
        return resized_img

    def postprocess(self, predictions):
        """
        6 point to sixdof post processing
        """
        output_rotate_matrix = compute_rotation_matrix_from_ortho6d(predictions)
        euler = compute_euler_angles_from_rotation_matrices(output_rotate_matrix)*180/np.pi
        pitch, yaw, roll = euler[:, 0], euler[:, 1], euler[:, 2]
        return pitch, yaw, roll
    

 
class SixDofEnd2End(object):
    def __init__(self, yolo_engine_path, dof_engine_path, logger, box_margin=10, stream_type='img',yolo_model_name='yolo', dof_model_name='dof', print_log=False):
        self.yolo_model = YOLOEngine(yolo_engine_path, logger, trt_nms=False, print_log=print_log) # dof model always using onnx model
        self.dof_model = DoFEngine(dof_engine_path, model_name=dof_model_name, logger=logger, print_log=print_log)
        self.logger = logger
        self.stream_type = stream_type
        self.box_margin = box_margin
    
    def forward(self, img):
        if self.stream_type == 'img' or self.stream_type == 'video':
            yolo_preproc_img = self.yolo_preprocess(img)
            h, w = yolo_preproc_img.shape[1:]
            detect_result, box_trigger = self.yolo_model.forward_yolo(yolo_preproc_img)
            if box_trigger:
                detect_box = detect_result[:4]
                detect_score = detect_result[4] * detect_result[5]
                detect_kp = detect_result[6:]
                x1, x2, y1, y2 = detect_box[0] - detect_box[2]/2, detect_box[0] + detect_box[2]/2, detect_box[1] - detect_box[3]/2, detect_box[1] + detect_box[3]/2
                face_img = yolo_preproc_img.transpose(1,2,0)[max(int(y1)-self.box_margin, 0):min(int(y2)+self.box_margin, h), max(int(x1)-self.box_margin, 0):min(int(x2)+self.box_margin, w), :]
                pitch, yaw, roll = self.dof_model.dof_forward(face_img)
                return yolo_preproc_img, detect_box, detect_score, detect_kp, face_img, pitch, yaw, roll
            else:
                return None

    def yolo_preprocess(self, img):
        img_copy = cv2.resize(img, self.yolo_model.imgsz)
        img_copy = img_copy / 255.0
        img_copy = img_copy[:,:,::-1].transpose(2,0,1)
        img_copy = np.ascontiguousarray(img_copy, dtype=np.float32)
        return img_copy

