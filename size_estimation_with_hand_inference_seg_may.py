# -*- coding: ISO-8859-1 -*-


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
import sys
import os
# from lang_sam import LangSAM

# 獲取當前檔案所在的資料夾路徑
current_path = os.path.dirname(os.path.abspath(__file__))

# 新增 FastSAM 資料夾到 sys.path
sys.path.append(os.path.join(current_path))

# Calculate Length utils
from utils.length_detection_utils import load_pose, depth_to_rgb_transit, estimate_length, filter, filter_v2, masked_image, get_length_dict

# hl2ss之模組
from hl2ss_API import hl2ss_3dcv

# model 所需模組以及相關程式碼
import torch
import warnings

torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.join(current_path, 'FastSAM'))

from FastSAM.fastsam import FastSAM, FastSAMPrompt
import time, gc, clip
import time


os.environ['TORCH_HOME'] = os.path.join('pretrained_model')


''' 全域變數 '''
# HoloLens 2 IP
HoloLens_IP = "192.168.172.131"
HoloLens_port = 12346

# 伺服器的ip以及監聽的port
server_IP = "192.168.172.75"
server_port = 12345

# 儲存深度感測器內部參數的路徑(必要)
# calibration_path = "./length_detection/data/0830/calibration_0830/"
# calibration_path = "./length_detection/data/0925/HOLOLENS-22JLGL/"

# RGB image設定(16:9)
RGB_width = 1920
RGB_height = 1080

# 暫存資料
RGB_images = None
RGB_focal_length = None
RGB_principal_point = None
RGB_pose = None

depth_images = None
depth_pose = None

tolerance_scope = 0

remove_backround_rgb = None  # 去背影像
max_length = -1

''' 全域變數 '''



def build_model(pretrainewd_path: str,
                img_size: int,
                fpn_size: int,
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True,
                use_selection: bool = True,
                use_combiner: bool = True,
                comb_proj_size: int = None):
    from models.pim_module.pim_module_eval import PluginMoodel

    model = \
        PluginMoodel(img_size=img_size,
                     use_fpn=use_fpn,
                     fpn_size=fpn_size,
                     proj_type="Linear",
                     upsample_type="Conv",
                     use_selection=use_selection,
                     num_classes=num_classes,
                     num_selects=num_selects,
                     use_combiner=use_combiner,
                     comb_proj_size=comb_proj_size,
                     isPretrained=False)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path, map_location='cpu')
        pretrained_dict = ckpt['model_state_dict']  # {k: v for k, v in ckpt['model_state_dict'].items() if
        # k in model.state_dict() and 'head' not in k}  # ('patch' in k or 'layer' in k or 'norm' in k)}
        model.load_state_dict(pretrained_dict, strict=True)

    model.eval()
    return model


@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
        ['layer1', 'layer2', 'layer3', 'layer4',
         'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4',
         'comb_outs']

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]

        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
            # pass
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out  # note that use '+=' would cause inplace error
    return sum_out / len(target_layer_names)


if __name__ == "__main__":
    # model
    # print("開始引入辨識模型")

    model_pt_path = "./records/FGVC-HERBS/baseline(ImageNet)_baseline_avg-20_max_M58-Total(104)_withoutbg_final_HERBS/save_model/M58-best.pth"
    pt_file = torch.load(model_pt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    length_dict = pt_file['real_length_dict']
    model = build_model(pretrainewd_path=model_pt_path,
                        img_size=pt_file['img_size'],
                        fpn_size=pt_file['fpn_size'],
                        num_classes=pt_file['num_class'],
                        num_selects=pt_file['num_selects'])
    # model.cpu()
    model.cuda()
    img_size = pt_file['img_size']
    class2num = pt_file['class2num']
    num2class = dict((value, key) for key, value in class2num.items())
    
    DEVICE = "cuda:0"if torch.cuda.is_available() else "cpu"
    seg = FastSAM('./FastSAM/weights/FastSAM-x_best_v2.pt')
    clip_model, preprocess = clip.load('ViT-B/32', device=DEVICE)
    # seg = LangSAM()
    
    hololens_names = ['8L7D4H','M1CPHO'] #,'M1CPHO'
    # hololens_names = ['9COGU3']
    date_folder =  '1017' #'0925' #
    # date_folder =  '1024' #'0925' #

    for hololens_name in hololens_names:
    # model
        calibration_path = f"./length_detection/data/{date_folder}/HOLOLENS-{hololens_name}/"
        calibration_lt = hl2ss_3dcv.get_calibration_rm(HoloLens_IP, 3805, calibration_path)  # 內外參是4*4的矩陣，而且是轉置矩陣
        uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, 320, 288)  # (288, 320, 2)
        xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy,
                                                    calibration_lt.scale)  # The calibration_lt.scale(1000) value is used to convert depth units to meters.

        RGB_intrinsics = np.eye(3, 4, dtype=np.float32)
        R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

        # 讀取資料

        # rgb_image_path = os.path.join('length_detection', 'data', '0830', 'RGB_small_test')
        # depth_image_path = os.path.join('length_detection', 'data', '0830', 'Depth')
        # img_info_csv = os.path.join('length_detection', 'data', '0830', 'data_830.xlsx')

        root_folder = f'{hololens_name}_output'
        # rgb_image_path = os.path.join('length_detection', 'data', date_folder, root_folder, 'RGB_test')
        rgb_image_path = os.path.join('length_detection', 'data', date_folder, root_folder, 'RGB')

        
        rgb_image_path2 = os.path.join('length_detection', 'data', 'RedBoxed')
        depth_image_path = os.path.join('length_detection', 'data', date_folder, root_folder, 'Depth')
        img_info_csv = os.path.join('length_detection', 'data', date_folder, root_folder, f'data_{hololens_name}.xlsx')
        record_out =  os.path.join('length_detection', 'records', date_folder+'_'+root_folder+"output_no_seg")

        if not os.path.exists(record_out):
            os.mkdir(record_out)

        saved_summary_path = os.path.join('length_detection', 'records')
        cls_folder = os.listdir(rgb_image_path)
        cls_folder.sort()

        # 儲存預測資訊
        pd_img_name = []
        pd_class_idx = []
        pd_class_name = []
        pd_measure_length = []
        pd_real_length = []

        pd_probs_list = []
        pd_preds_list = []
        pd_ld_probs_list = []
        pd_ld_preds_list = []
        pd_class_name_list = {}
        pd_total_class_img_num_dict = {}
        measure_avg_list = []
        measure_std_list = []
        real_lengths =[]

        red_boxes_list =[]

        top1_dic, top5_dic = {}, {}
        ld_top1_dic, ld_top5_dic = {}, {}
        top1_acc, top5_acc = 0, 0
        ld_top1, ld_top5 = 0, 0


        # Second Stage classes id
        need_scond_stage = [92, 82, 38, 41, 44, 51, 86, 87, 88, 40, 48, 49, 103] ### 端板群、套管群


        is_save_summary = True


        pbar = tqdm.tqdm(total=len(cls_folder), ascii=True)
        update_n = 0 

        # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'masked')):
        #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'masked'))
        # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'segmentation')):
        #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'segmentation'))
        # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'contour')):
        #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'contour'))
        length_dict = get_length_dict(os.path.join('length_detection','data','M58-Length.txt'))

        print(cls_folder)
        for ci, cf in enumerate(cls_folder):
            # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'masked', cf)):
            #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'masked', cf))
            # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'segmentation', cf)):
            #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'segmentation', cf))
            # if not os.path.exists(os.path.join('length_detection', 'data', date_folder, root_folder, 'contour', cf)):
            #     os.mkdir(os.path.join('length_detection', 'data', date_folder, root_folder, 'contour', cf))
            # if not os.path.exists(os.path.join('length_detection', 'data', 'RedBoxed', 'segmentation', cf)):
            #     os.mkdir(os.path.join('length_detection', 'data', 'RedBoxed', 'segmentation', cf))


            if not os.path.exists(os.path.join(rgb_image_path2, cf)):
                print('red boxes not found')
                continue
            print(cf)
            update_n += 1

            images_list = os.listdir(os.path.join(rgb_image_path, cf))
            images_list.sort()

            class_name = cf
            classes_measure_list = []
            classes_std_list = []
            print(class_name)
            pd_class_name_list[class2num[class_name]] = class_name
            top1_dic[class2num[class_name]] = top1_dic.get(class2num[class_name], 0)
            ld_top1_dic[class2num[class_name]] = ld_top1_dic.get(class2num[class_name], 0)
            top5_dic[class2num[class_name]] = top5_dic.get(class2num[class_name], 0)
            ld_top5_dic[class2num[class_name]] = ld_top5_dic.get(class2num[class_name], 0)
            pd_real_length.append(length_dict[class_name])
            print(images_list)
            for counter, images in enumerate(images_list):

                # try:
                real_lengths.append(length_dict[class_name])
                print(images)
                start_time = time.time()

                # record predict result
                pd_img_name.append(images)
                pd_class_idx.append(class2num[cf])
                pd_class_name.append(cf) 
                pd_total_class_img_num_dict[class2num[class_name]] = 1 + pd_total_class_img_num_dict.get(
                    class2num[class_name], 0)

                img_choose = images.split('_')[1]
                img_choose = int(img_choose.split('.')[0])
                # img_choose = images.split('.')[0]


                # if img_choose >= 126:
                #     img_info_csv = os.path.join('length_detection', 'data', '0731','data_731_2.xlsx')
                # else:
                #     img_info_csv = os.path.join('length_detection', 'data', '0731','data_731_1.xlsx')

                '''Load Image Info'''
                fl_data = load_pose(img_info_csv, 'fl', img_choose)
                x, y = fl_data[0][0], fl_data[0][1]
                tmp = [x, y]
                RGB_focal_length = tmp
                

                pp_data = load_pose(img_info_csv, 'pp', img_choose)
                x, y = pp_data[0][0], pp_data[0][1]
                tmp = [x, y]
                RGB_principal_point = tmp
                

                rgb_pose = load_pose(img_info_csv, 'RGB_pose', img_choose)
                rgb_pose = rgb_pose.to_numpy()
                RGB_pose = rgb_pose
                

                depth_pose = load_pose(img_info_csv, 'Depth_pose', img_choose)
                depth_pose = depth_pose.to_numpy()

                '''Load RGB/Depth Image'''
                # print( os.listdir(os.path.join(rgb_image_path2, cf)))
                try:
                    images2 = os.listdir(os.path.join(rgb_image_path2, cf))[counter]
                except:
                    a = len(os.listdir(os.path.join(rgb_image_path2, cf)))
                    images2 = os.listdir(os.path.join(rgb_image_path2, cf))[counter%a]
                red_boxes_list.append(images2)
                
                bgr = cv2.imdecode(np.fromfile(os.path.join(rgb_image_path, cf, images), dtype=np.uint8), -1)
                bgr2 = cv2.imdecode(np.fromfile(os.path.join(rgb_image_path2, cf, images2), dtype=np.uint8), -1)
                # bgr = cv2.imread(os.path.join(rgb_image_path, cf, images), cv2.IMREAD_COLOR)
                rgb = bgr[:, :, ::-1]
                rgb2 = bgr2[:, :, ::-1]
                # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                RGB_images = bgr
                RGB_images2 = bgr2

                RGB_masked_images = masked_image(RGB_images)
                # cv2.imwrite(os.path.join('length_detection', 'data', date_folder, root_folder, 'masked', cf, images), RGB_masked_images)
                RGB_masked_images2 = masked_image(RGB_images2)

                depth_images = cv2.imdecode(
                    np.fromfile(os.path.join(depth_image_path, f'{hololens_name}-Depth_{img_choose}.png'), dtype=np.uint8), -1)
                # depth_images = cv2.imread(depth_image_path, f'Depth_{img_choose}.png', cv2.IMREAD_UNCHANGED)
                depth_images = depth_images / 1000 #(288,320)
                # print(depth_images.shape)
                depth_images = np.expand_dims(depth_images, axis=-1) #(288,320,1)

                ''' 座標轉換過程 '''

                depth_to_rgb_mapping_matrix = depth_to_rgb_transit(calibration_lt, depth_images, 
                                                                xy1, RGB_focal_length, RGB_principal_point, RGB_pose, depth_pose)

                ''' 座標轉換過程 '''


                '''Remove background'''
                text_prompt = 'industrial component'
                everything_results = seg(RGB_masked_images, device=DEVICE, retina_masks=True,  conf=0.4, iou=0.8,)
                prompt_process= FastSAMPrompt(RGB_masked_images, everything_results, device=DEVICE)
                ann, source = prompt_process.text_prompt_custom(text_prompt, clip_model, preprocess)
                out = os.path.join('length_detection', 'data', date_folder, root_folder, 'contour', cf, images)
                # final_image = None
                if prompt_process.results != None:
                    ann = ann.reshape(ann.shape[1],ann.shape[2])
                    source[~ann] = 0
                    final_image = source
                    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(os.path.join('length_detection', 'data', date_folder, root_folder, 'segmentation', cf, images), final_image)
                    length = estimate_length(depth_to_rgb_mapping_matrix, calibration_lt, final_image, depth_images,out)
                else :
                    length = 0
                #Langsam
                """text_prompt = "chip, metal"
                RGB_images = Image.fromarray(RGB_masked_images)
                masks, boxes, phrases, logits = seg.predict(RGB_images, text_prompt)
                masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
                boxes = [box.squeeze().cpu().numpy() for box in boxes]

                expanded_mask = np.expand_dims(masks_np[0], axis=-1)
                expanded_mask = np.repeat(expanded_mask, 3, axis=-1)
                masked_image_ = np.where(expanded_mask, RGB_images, 0)
                final_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
                final_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                final_image = masked_image_"""
                

            
                if length >= 0.2659:
                    tolerance_scope = 0.045
                else:
                    tolerance_scope = 0.015

                classes_measure_list.append(length)
                pd_measure_length.append(length)

                # model   
                # Modify add mixed information        
                transform = A.Compose(
                    [
                        # A.Crop(x_min=604, y_min=186, x_max=1314, y_max=896), #crop center 
                        A.Crop(1000, 186, 1710, 896),#crop left
                        # A.Crop(xmin, ymin, xmax, ymax),
                        A.Resize(img_size, img_size),
                        A.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])
                
                everything_results2 = seg(RGB_masked_images2, device=DEVICE, retina_masks=True,  conf=0.4, iou=0.8,)
                prompt_process2 = FastSAMPrompt(RGB_masked_images2, everything_results2, device=DEVICE)
                ann2, source2 = prompt_process2.text_prompt_custom(text_prompt, clip_model, preprocess)

                if prompt_process2.results != None:
                    ann2 = ann2.reshape(ann2.shape[1],ann2.shape[2])
                    
                    source2[~ann2] = 0
                    final_image2 = source2
                    final_image2 = cv2.cvtColor(final_image2, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(os.path.join('length_detection', 'data', 'RedBoxed', 'segmentation', cf, images), final_image)
                #Langsam
                """text_prompt = "chip, metal"
                RGB_images2 = Image.fromarray(RGB_masked_images2)
                masks2, boxes2, phrases2, logits2 = seg.predict(RGB_images2, text_prompt)
                masks_np2 = [mask.squeeze().cpu().numpy() for mask in masks2]
                boxes2 = [box.squeeze().cpu().numpy() for box in boxes2]

                expanded_mask = np.expand_dims(masks_np2[0], axis=-1)
                expanded_mask = np.repeat(expanded_mask, 3, axis=-1)
                masked_image_ = np.where(expanded_mask, RGB_images, 0)
                final_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)"""
                #final_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                # final_image = masked_image_
                # preprocessed_img = transform(image=final_image2)['image']  # 直接用segment anything後的圖片去crop,偵測
                preprocessed_img = transform(image=RGB_masked_images2)['image']  # 直接用segment anything後的圖片去crop,偵測

                # except:
                #         preprocessed_img = transform(image=RGB_images2)['image']  # 直接用segment anything後的圖片去crop,偵測
    
                preprocessed_imgs = preprocessed_img.unsqueeze(0).cuda()  # 增加一個維度在前面，用於一批次的數量。

                with torch.no_grad():
                    preprocessed_imgs = preprocessed_imgs.cuda()
                    outs = model.forward(preprocessed_imgs)
                    sum_outs = sum_all_out(outs, sum_type="softmax")
                    probs, preds = torch.sort(sum_outs, dim=-1,
                                            descending=True)  # 最高概率的排前面，preds[0][0]是最終預測結果(torch.Size([1, 50]))
                    
                    # length_dict = pt_file['real_length_dict']
                    length_dict = get_length_dict(os.path.join('length_detection','data','M58-Length.txt'))
                    # Update length dict
                    # length_dict.update({"M58-從動輪軸心端板": 0.02199})
                    # length_dict.update({"M58-走行驅動軸心": 1.366})

                    ''' 實驗:未使用尺寸資訊 '''
                    pd_probs_list.append(probs[0].cpu())
                    pd_preds_list.append(preds[0].cpu())
                    if class2num[class_name] in preds[0][:1]:
                        top1_acc += 1
                        if is_save_summary:
                            top1_dic[class2num[class_name]] = 1 + top1_dic.get(class2num[class_name], 0)
                    if class2num[class_name] in preds[0][:5]:
                        top5_acc += 1
                        if is_save_summary:
                            top5_dic[class2num[class_name]] = 1 + top5_dic.get(class2num[class_name], 0)

                    ''' 實驗:使用尺寸資訊 '''
                    if length !=0:
                        adjusted_preds, adjusted_probs = filter(length_dict, preds, probs, num2class, tolerance_scope, length,10)
                    else:
                        adjusted_preds, adjusted_probs = preds, probs
                    
                    # tmp_adjusted_preds = list(adjusted_preds[0].cpu())
                    # if tmp_adjusted_preds[0] in need_scond_stage:
                    #     tolerance_scope = 0.008
                    #     adjusted_preds, adjusted_probs = filter(pt_file['real_length_dict'], adjusted_preds, adjusted_probs, num2class, tolerance_scope, length)
                    
                    pd_ld_probs_list.append(adjusted_probs[0].cpu())
                    pd_ld_preds_list.append(adjusted_preds[0].cpu())
                    if class2num[class_name] in adjusted_preds[0][:1]:
                        ld_top1 += 1
                        if is_save_summary:
                            ld_top1_dic[class2num[class_name]] = 1 + ld_top1_dic.get(class2num[class_name], 0)
                    if class2num[class_name] in adjusted_preds[0][:5]:
                        ld_top5 += 1
                        if is_save_summary:
                            ld_top5_dic[class2num[class_name]] = 1 + ld_top5_dic.get(class2num[class_name], 0)
                # except:
                #     print(cf, images,'failed')
                #     continue
            measure_avg_list.append(np.array(classes_measure_list).mean())
            measure_std_list.append(np.array(classes_measure_list).std())
                    
            pbar.update(update_n)
            update_n = 0
        pbar.close()

        if is_save_summary:
            print(len(pd_class_idx), len(pd_class_name), len(pd_img_name), len(pd_measure_length))
            df = pd.DataFrame(
                {'class id': pd_class_idx, 'class name': pd_class_name, 'img name': pd_img_name,
                'length estimate': pd_measure_length, 'real_length':real_lengths})
            df.to_excel(os.path.join(record_out, 'length_estimate.xlsx'))

            df = pd.DataFrame(
                {'class id': pd_class_idx, 'class name': pd_class_name,
                'redboxes':red_boxes_list, 'img name': pd_img_name, 'ori_preds': pd_preds_list,
                'ori_probs': pd_probs_list, 'ld_preds': pd_ld_preds_list, 'ld_probs': pd_ld_probs_list})
            df.to_excel(os.path.join(record_out, 'output.xlsx'))

            df = pd.DataFrame({
                'class id': top1_dic.keys(), 'class name': pd_class_name_list.values(),'real length':pd_real_length,
                'classes estimate average': measure_avg_list, 'classes estimate std': measure_std_list
            })
            df.to_excel(os.path.join(record_out, 'length_estimate_info.xlsx'))

            top1_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in
                            zip(top1_dic.values(), pd_total_class_img_num_dict.values())]
            top5_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in
                            zip(top5_dic.values(), pd_total_class_img_num_dict.values())]
            
            df = pd.DataFrame(
                {'class id': top1_dic.keys(), 'class name': pd_class_name_list.values(),
                'total num img': pd_total_class_img_num_dict.values(), 'top1_correct': top1_dic.values(),
                'top1_acc': top1_acc_list, 'top5_correct': top5_dic.values(), 'top5_acc': top5_acc_list, })
            df.to_excel(os.path.join(record_out, 'without_ld_output.xlsx'))

            top1_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in
                            zip(ld_top1_dic.values(), pd_total_class_img_num_dict.values())]
            top5_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in
                            zip(ld_top5_dic.values(), pd_total_class_img_num_dict.values())]
            df = pd.DataFrame(
                {'class id': ld_top1_dic.keys(), 'class name': pd_class_name_list.values(),
                'total num img': pd_total_class_img_num_dict.values(), 'top1_correct': ld_top1_dic.values(),
                'top1_acc': top1_acc_list, 'top5_correct': ld_top5_dic.values(), 'top5_acc': top5_acc_list, })
            df.to_excel(os.path.join(record_out, 'ld_output.xlsx'))
