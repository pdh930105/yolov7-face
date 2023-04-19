import numpy as np
import cv2
import matplotlib.pyplot as plt

import pyrealsense2 as rs

from math import cos, sin

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = np.sqrt(np.power(v, 2).sum(1))# batch
    eps = np.array([1e-8], dtype=np.float32)
    v_mag = np.maximum(v_mag, eps)
    v_mag = v_mag.reshape(batch, 1)
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = np.concatenate((i.reshape(batch,1), j.reshape(batch,1), k.reshape(batch,1)),1) #batch*3
        
    return out
        

def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
    x = x.reshape(-1,3,1)
    y = y.reshape(-1,3,1)
    z = z.reshape(-1,3,1)
    matrix = np.concatenate((x,y,z), 2) #batch*3*3
    return matrix

def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = np.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = np.array(singular, dtype=np.float32)    
    x = np.arctan2(R[:,2,1], R[:,2,2])
    y = np.arctan2(-R[:,2,0], sy)
    z = np.arctan2(R[:,1,0],R[:,0,0])
    
    xs = np.arctan2(-R[:,1,2], R[:,1,1])
    ys = np.arctan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
    
    out_euler = np.zeros((batch, 3))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    """
    Prints the person's name and age.

    If the argument 'additional' is passed, then it is appended after the main info.

    Parameters
    ----------
    img : array
        Target image to be drawn on
    yaw : int
        yaw rotation
    pitch: int
        pitch rotation
    roll: int
        roll rotation
    tdx : int , optional
        shift on x axis
    tdy : int , optional
        shift on y axis
        
    Returns
    -------
    img : array
    """

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

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

def vis_end2end(img, boxes, scores, cls_ids, conf=0.5, class_names=None, box_type='xyxy'):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(0)
        score = scores[i]
        if score < conf:
            continue
        if box_type =='xyxy':
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
        else: # using xywh
            x0 = int(box[0]-box[2]/2)
            x1 = int(box[0]+box[2]/2)
            y0 = int(box[1]-box[3]/2)
            y1 = int(box[1]+box[3]/2)

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