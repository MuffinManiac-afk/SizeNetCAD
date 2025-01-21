import numpy as np
import cv2
from PIL import Image
from scipy.cluster.vq import kmeans2
from matplotlib import pyplot as plt
import av
import socket
import struct
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
import pandas as pd
import tqdm

# hl2ss之模組
from hl2ss_API import hl2ss_3dcv
import os

# model 所需模組以及相關程式碼
import torch
import warnings
# from lang_sam import LangSAM

torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['TORCH_HOME'] = os.path.join('pretrained_model')

def Create_RGB_Intrinsics(intrinsics, focal_length, principal_point):
    intrinsics[0, 0] = focal_length[0]
    intrinsics[1, 1] = focal_length[1]
    intrinsics[0, 2] = principal_point[0]
    intrinsics[1, 2] = principal_point[1]
    return intrinsics


def Calculate_Size(depth_images, object_mask, obj_w, obj_h, calibration_lt):
    sorted_object_depth_image = np.sort(depth_images[object_mask==255])
    distance = np.min(sorted_object_depth_image)
    e_width = (obj_w) * distance / calibration_lt.intrinsics[0, 0]
    e_height = (obj_h) * distance / calibration_lt.intrinsics[1, 1]
    return e_width, e_height

def check_valid_pixel_in_depth(depth_image, depth_x, depth_y): # check 7*7內有無有效像素
    if depth_image[depth_y, depth_x]==0: # 無效像素
        for row in range(depth_y-3, depth_y+4):
            for col in range(depth_x-3, depth_x+4):
                if (depth_y!=row or depth_x!=col) and depth_image[row, col]!=0:
                    return col, row
    
    return depth_x, depth_y

def find_nearest(depth_coords, target_coord):
    distances = np.sqrt((depth_coords[:, :, 0] - target_coord[0]) ** 2 + (depth_coords[:, :, 1] - target_coord[1]) ** 2)
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    return min_index


def length_detection(RGB_image, RGB_focal_length, RGB_principal_point, RGB_pose, Depth_image, Depth_pose, calibration_lt):

    # calibration_lt = hl2ss_3dcv.get_calibration_rm(HoloLens_IP, 3805, calibration_path)  # 內外參是4*4的矩陣，而且是轉置矩陣
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, 320, 288)  # (288, 320, 2)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy,
                                                  calibration_lt.scale)  # The calibration_lt.scale(1000) value is used to convert depth units to meters.

    RGB_intrinsics = np.eye(3, 4, dtype=np.float32)
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    

    RGB_intrinsics = Create_RGB_Intrinsics(RGB_intrinsics, RGB_focal_length, RGB_principal_point)
    depth_camera_points = xy1 * Depth_image  # (288,320,3) # 深度感測器:像素座標系轉換至相機座標系
    homogenous_depth_camera_points = np.append(depth_camera_points, np.ones((288, 320, 1)),
                                                axis=2)  # (288,320,4):變齊次

    # reshape(-1, 4)：這部分將數據重新排列成一個形狀為 (n, 4) 的矩陣，其中 n 是根據數據的總長度自動計算出來的。-1 表示自動計算這個維度的大小。
    # .T：這是轉置操作，將矩陣的行和列互換。
    homogenous_depth_camera_points_reshaped = homogenous_depth_camera_points.reshape(-1, 4).T  # (4, 92160)

    # 深度感測器之相機座標系轉換至世界座標系所需的矩陣
    depth_camera_to_world_matrix = np.transpose(Depth_pose) @ np.transpose(
        np.linalg.inv(calibration_lt.extrinsics))

    world_points = np.dot(depth_camera_to_world_matrix,
                            homogenous_depth_camera_points_reshaped)  # (4, 92160) : 世界座標

    # 世界座標轉換至RGB像素座標之矩陣 R:旋轉矩陣
    world_to_rgb_pixel_matrix = RGB_intrinsics @ R @ np.transpose(np.linalg.inv(RGB_pose))

    # 世界座標轉換至RGB像素座標
    depth_to_rgb_mapping_matrix = np.dot(world_to_rgb_pixel_matrix,
                                            world_points)  # (3, 92160)，但含有Zc，且為齊次(Zc[u,v,1])

    depth_to_rgb_mapping_matrix /= depth_to_rgb_mapping_matrix[-1, :]  # (3, 92160)，除上Zc，仍是齊次
    depth_to_rgb_mapping_matrix = depth_to_rgb_mapping_matrix[:2, :]  # (2, 92160)，消除齊次([u,v])
    depth_to_rgb_mapping_matrix = depth_to_rgb_mapping_matrix.T.reshape(288, 320, 2)  # 變成(288,320,2)

    try:
        # 根據segment anything產生的圖片去框出零件位置
        gray = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blurred, 30, 60)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edged = cv2.dilate(edged, kernel, iterations=2)  # 膨脹操作
        edged = cv2.erode(edged, kernel, iterations=2)  # 侵蝕操作

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出影像中所有最外圍的輪廓 ; ok
        main_contour = max(contours, key=cv2.contourArea)  # 取這些輪廓中面積最大的 ; ok
        minRect = cv2.minAreaRect(main_contour)  # 根據這個輪廓，找到最小矩形，主要用於估算長度
        box = cv2.boxPoints(minRect)  #回傳矩形的四點座標
        left_top = min(box[:, 0]), min(box[:, 1])
        right_bottom = max(box[:, 0]), max(box[:, 1])

        # RGB矩形頂點找到深度對應的地方
        lt_y, lt_x = find_nearest(depth_to_rgb_mapping_matrix, left_top)
        rb_y, rb_x = find_nearest(depth_to_rgb_mapping_matrix, right_bottom)

        box0_y, box0_x = find_nearest(depth_to_rgb_mapping_matrix, box[0])
        box1_y, box1_x = find_nearest(depth_to_rgb_mapping_matrix, box[1])
        box2_y, box2_x = find_nearest(depth_to_rgb_mapping_matrix, box[2])
        box3_y, box3_x = find_nearest(depth_to_rgb_mapping_matrix, box[3])

        # 深度影像中物體佔的像素寬度和長度(pixel)
        depth_width0 = np.sqrt((box0_x - box1_x) ** 2 + (box0_y - box1_y) ** 2)  # g1
        depth_height0 = np.sqrt((box1_x - box2_x) ** 2 + (box1_y - box2_y) ** 2)  # g2
        depth_width1 = np.sqrt((box2_x - box3_x) ** 2 + (box2_y - box3_y) ** 2)  # g1
        depth_height1 = np.sqrt((box3_x - box0_x) ** 2 + (box3_y - box0_y) ** 2)  # g2

        depth_obj_w = min(depth_width0, depth_width1)
        depth_obj_h = min(depth_height0, depth_height1)

        # 深度影像的處理
        depth_images = depth_images[:, :, 0]
        depth_images = depth_images * 100  # 公尺變公分

        # 估算物體大小(求最大長度,如無法求出,則返回-1)
        obj_width, obj_height = Calculate_Size(depth_images, lt_x, lt_y, rb_x, rb_y, depth_obj_w, depth_obj_h, calibration_lt)
        max_length = max(obj_width, obj_height) / 100

        return True, max_length, depth_images
    
    except Exception as e:
        return False