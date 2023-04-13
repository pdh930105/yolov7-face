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

class BaseEngine(object):
    def __init__(self, engine_path, logger, print_log=False):
        self.mean = None
        self.std = None
        self.n_classes = 1
        self.nkpt = 5
        self.use_onnx_trt = False
        self.class_names = ['face']
        self.logger =logger
        self.print_log = print_log

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        print("self.imgsz :", self.imgsz)
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

    def inference(self, img_path, conf=0.5, end2end=False):
        start_time = time.perf_counter()
        origin_img = cv2.imread(img_path)
        #img, ratio = preproc_pad(origin_img, self.imgsz, self.mean, self.std)
        #data = self.infer(img)
        
        if end2end:
            prepare_img_time = time.perf_counter()
            img, ratio = preproc_pad(origin_img, self.imgsz, self.mean, self.std)
            preprocess_img_time = time.perf_counter()
            data = self.infer(img)
            infer_time = time.perf_counter()
            if self.use_onnx_trt:
                raise
            else:
                num, final_boxes, final_scores, final_cls_inds = data

                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

                if dets is not None:
                    final_boxes, final_scores, final_classes = dets[:,:4], dets[:, 4], dets[:, 5]
                    origin_img = vis_end2end(origin_img, final_boxes, final_scores, final_classes,
                                    conf=conf, class_names=self.class_names)
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
                    

        else:
            resized_img, resized_img_tran = preproc(origin_img, self.imgsz)
            data = self.infer(resized_img_tran)
            #predictions = np.reshape(data, (1, -1, int(5+self.n_classes + self.nkpt*3)))[0] # does not using batch inference
            #dets = self.postprocess(predictions,ratio)
            #self.logger.info(f'data shape : {data.shape}')
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes + self.nkpt*3))) # does not using batch inference        
            self.logger.info("post process start")
            dets = self.postprocess_ops_nms(predictions=predictions)[0]
            #print("output", len(dets), print(dets[0]))
            if dets is not None:
                final_boxes, final_scores, final_key_points = dets[:,:4], dets[:, 4], dets[:, 5:]
                origin_img = vis(resized_img, final_boxes, final_scores, final_key_points,
                                conf=conf, class_names=self.class_names)
            if self.print_log:
                print(len(data))
                print(predictions[:, :, 4].max())
                print(predictions[:, :, 5].max())
                
        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        keypoints = predictions[:, 6:]
        scores = predictions[:,4:5] # * predictions[:,5:6] # obj prod * class prod (face)
        scores_mask = scores > 0.1
        scores_mask = scores_mask.squeeze()
        if scores_mask.sum() == 0:
            return None       

        boxes = boxes[scores_mask, :]
        keypoints = keypoints[scores_mask, :]
        scores = scores[scores_mask, :]        
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        keep = nms(boxes_xyxy, scores, nms_thr=0.45)
        dets_boxes = boxes_xyxy[keep]
        dets_scores = scores[keep]
        dets_keypoint = keypoints[keep]
        num_idx = np.zeros([len(dets_boxes), 1])
        dets = np.concatenate([dets_boxes, dets_scores, dets_keypoint], 1)
        return dets
    
    @staticmethod
    def postprocess_ops_nms(predictions, conf_thres =0.25, iou_thres=0.45, classes=None):
        kpt_label=5
        min_wh, max_wh = 2, 4096
        prediction = torch.from_numpy(predictions)
        xc = prediction[..., 4] > conf_thres
        output = [torch.zeros((0, kpt_label*3+6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # Compute conf
            cx, cy, w, h = x[:,0:1], x[:,1:2], x[:,2:3], x[:,3:4]
            obj_conf = x[:, 4:5]
            #print("obj_conf :", obj_conf)
            cls_conf = x[:, 5:6]
            #print("cls_conf :", cls_conf)
            kpts = x[:, 6:]
            cls_conf = obj_conf * cls_conf  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            x_min = cx - (w/2)
            y_min = cy - (h/2)
            x_max = cx + (w/2)
            y_max = cy + (h/2)
            box = torch.cat((x_min, y_min, x_max, y_max), 1)            
            conf, j = cls_conf.max(1, keepdim=True)
            #print("after class conf :", conf)
            x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]
            #print("x shape : ", x.shape)
            c = x[:, 5:6] * 0
            #print("c shape :", c)
            boxes, scores = x[:, :4] +c , x[:, 4]  # boxes (offset by class), scores
            #print("boxes.shape :", boxes.shape)
            #print("box value :", boxes, scores)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            output[xi] = x[i]
        return output
        
    def get_fps(self):
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        t1 = time.perf_counter()
        print(f"yolo model inference (without preprocess, postprocess)\n{(t1 - t0)/100:.3f}s ({100/(t1 - t0):.3f} FPS)")
    
    def detect_video(self, video_path, use_cam=True,conf=0.5, end2end=False):
        if use_cam:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FRAME_COUNT, 30)
        else:
            cap = cv2.VideoCapture(video_path)
          
        #out = cv2.VideoWriter('./001.avi',fourcc,fps,(width,height))
        fps = 0
        import time
        if end2end:
            img_preprocess_list = []
            img_read_list = []
            img_resize_list = []
            infer_process_list = []
            post_process_list = []
            total_process_list = []
            while True:
                frame_pre_process_time = time.perf_counter()
                ret, frame = cap.read()
                frame_read_process_time = time.perf_counter()
                if not ret:
                    break
                if not use_cam:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                frame_resize_process_time = time.perf_counter()

                blob, ratio = preproc_pad(frame, self.imgsz, self.mean, self.std)
                frame_post_process_time = time.perf_counter()
                img_read_list.append(frame_read_process_time - frame_pre_process_time)
                img_resize_list.append(frame_resize_process_time - frame_read_process_time)
                img_preprocess_list.append(frame_post_process_time-frame_pre_process_time)
                
                data = self.infer(blob)
                infer_time = time.perf_counter()
                infer_process_list.append(infer_time-frame_post_process_time)


                fps = (fps + (1. / (time.time() - frame_pre_process_time))) / 2
                resized_img = cv2.putText(blob, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
            
                num, final_boxes, final_scores, final_cls_inds = data
                self.logger.info(f'detected face : {num}')
                self.logger.info(f'boxes info : {final_boxes[:num[0]*4]}')
                self.logger.info(f'predict : {final_scores[:num[0]]*100}%')
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:,
                                                                    :4], dets[:, 4], dets[:, 5]
                    frame = vis_end2end(resized_img, final_boxes, final_scores, final_cls_inds,
                                    conf=conf, class_names=self.class_names)
                    vis_frame = frame[::-1,:, :].transpose([1,2,0])
                    #cv2.imshow('frame', vis_frame)
                #out.write(frame)
                post_process_time = time.perf_counter()
                post_process_list.append(post_process_time-infer_time)
                total_process_list.append(post_process_time - frame_pre_process_time)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            #out.release()
            cap.release()
            cv2.destroyAllWindows()
            if self.print_log:
                print("total frame : ", len(post_process_list))
                print(f"img read average : {np.array(img_read_list).sum() / len(img_preprocess_list):.3f} s (max {max(img_read_list):.3f}, min {min(img_read_list):.3f})")
                print(f"img resize average : {np.array(img_resize_list).sum() / len(img_preprocess_list):.3f} s (max {max(img_resize_list):.3f}, min {min(img_resize_list):.3f})")
                print(f"total preprocess average : {np.array(img_preprocess_list).sum() / len(img_preprocess_list):.3f} s (max {max(img_preprocess_list):.3f}, min {min(img_preprocess_list):.3f})")
                print(f"inference average : {np.array(infer_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(infer_process_list):.3f}, min {min(infer_process_list):.3f})")
                print(f"postprocess average : {np.array(post_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(post_process_list):.3f}, min {min(post_process_list):.3f})")
                print(f"total process average : {np.array(total_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(total_process_list):.3f}, min {min(total_process_list):.3f})")

        else:
            self.logger.info("post process start")
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes+ self.nkpt*3)))[0]
            dets = self.postprocess_ops_ns(predictions)[0]

    def detect_video_v2(self, video_path, use_cam=True,conf=0.5, end2end=False):
        
        #out = cv2.VideoWriter('./001.avi',fourcc,fps,(width,height))
        import time
        video_stream = LoadImages(video_path, self.imgsz[0])
        fps=0
        if end2end:
            img_preprocess_list = []
            infer_process_list = []
            post_process_list = []
            total_process_list = []
            post_process_time = time.perf_counter()
            start_time = post_process_time
            for source, img, img0, cap in video_stream:
                frame_pre_process_time = time.perf_counter()
                blob = img
                img_preprocess_list.append(frame_pre_process_time-post_process_time)
                data = self.infer(blob)
                infer_time = time.perf_counter()
                infer_process_list.append(infer_time-frame_pre_process_time)

                fps = (fps + (1. / (time.time() - frame_pre_process_time))) / 2
                resized_img = cv2.putText(blob, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
            
                num, final_boxes, final_scores, final_cls_inds = data
                #self.logger.info(f'detected face : {num}')
                #self.logger.info(f'boxes info : {final_boxes[:num[0]*4]}')
                #self.logger.info(f'predict : {final_scores[:num[0]]*100}%')
                #final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                final_boxes = final_boxes.reshape(-1, 4)
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:,
                                                                    :4], dets[:, 4], dets[:, 5]
                    frame = vis_end2end(resized_img, final_boxes, final_scores, final_cls_inds,
                                    conf=conf, class_names=self.class_names)
                    vis_frame = frame[::-1,:, :].transpose([1,2,0])
                    #cv2.imshow('frame', vis_frame)
                #out.write(frame)
                post_process_time = time.perf_counter()
                post_process_list.append(post_process_time-infer_time)
                total_process_list.append(post_process_time - start_time)
                start_time = time.perf_counter()
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            #out.release()
            cap.release()
            cv2.destroyAllWindows()
            if self.print_log:
                print("total frame : ", len(post_process_list))
                print(f"preprocess average : {np.array(img_preprocess_list).sum() / len(img_preprocess_list):.3f} s (max {max(img_preprocess_list):.3f}, min {min(img_preprocess_list):.3f})")
                print(f"inference average : {np.array(infer_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(infer_process_list):.3f}, min {min(infer_process_list):.3f})")
                print(f"postprocess average : {np.array(post_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(post_process_list):.3f}, min {min(post_process_list):.3f})")
                print(f"total process average : {np.array(total_process_list).sum() / len(img_preprocess_list):.3f} s (max {max(total_process_list):.3f}, min {min(total_process_list):.3f})")
        else:
            self.logger.info("post process start")
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes+ self.nkpt*3)))[0]
            dets = self.postprocess_ops_nms(predictions)[0]


    def detect_rs(self, rs_type, conf=0.5):
        rs_stream = LoadRealSense(rs_type, img_size=self.imgsz[0], stride=32)
        t0 = time.perf_counter()
        img_preprocess_list = []
        infer_process_list = []

        for source, img, img0, depth_img, depth_img0, _ in rs_stream:
            t1 = time.perf_counter()
            data = self.infer(img)
            t2 = time.perf_counter()
            img_preprocess_list.append(t1 - t0)
            infer_process_list.append(t2-t1)
            t0 = t2
            num, final_boxes, final_scores, final_cls_inds = data
            if int(num) > 0:
                self.logger.info(f'detected face : {num}')
                self.logger.info(f'boxes info : {final_boxes[:num[0]*4]}')
                self.logger.info(f'predict : {final_scores[:num[0]]*100}%')
                

        print(f"total frame : {len(img_preprocess_list)}")
        print(f"preprocess average : {np.array(img_preprocess_list).sum() / len(img_preprocess_list):.3f} s (max {max(img_preprocess_list):.3f}, min {min(img_preprocess_list):.3f})")
        print(f"infer process average : {np.array(img_preprocess_list).sum() / len(infer_process_list):.3f} s (max {max(infer_process_list):.3f}, min {min(infer_process_list):.3f})")



             
        
def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def preproc(image, input_size, swap=(2,0,1)):
    resized_img = cv2.resize(image, input_size)
    #resized_img = resized_img / 255
    resized_img_transpose = resized_img.transpose(swap)
    resized_img_transpose = resized_img_transpose / 255
    resized_img_transpose = np.ascontiguousarray(resized_img_transpose, dtype=np.float32)
    return resized_img, resized_img_transpose

def preproc_pad(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1] # bgr to rgb
    padded_img /= 255.0 # normalize
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap) # h, w, c to c, h, w
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) # stream for cuda
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)

def vis_end2end(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(0)
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score*100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis(img, boxes, scores, keypoints, conf=0.1, class_names=None):
    #img = cv2.resize(img, (640, 640))
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(0)
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis_keypoint(img, boxes, scores, keypoints, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        keypoint = keypoints[i]
        cls_id = 0
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img