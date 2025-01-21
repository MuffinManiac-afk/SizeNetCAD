import torch, random
import warnings

torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import tqdm
from vis_utils import test_ImgLoader
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from eval import count_total_pick_times, avg_result, choose_random_paths, length_detection, length_detection_withReal
import time
from models.classification_model import clf_model
os.environ['TORCH_HOME'] = os.path.join('pretrained_model')


def build_model(pretrainewd_path: str,
                img_size: int,
                fpn_size: int,
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True,
                use_selection: bool = True,
                use_combiner: bool = True,
                comb_proj_size: int = None):
    """
    create HERBS model
    """
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
        ckpt = torch.load(pretrainewd_path)
        pretrained_dict =  ckpt['model_state_dict']
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


def save_confusion_matrix(y_true, y_pred, class2num, output_path,type='after LD',model_name=''):
    """
    output confusion matrix
    """
    font = {'family': 'Microsoft YaHei',
            'weight': 'bold',
            'size': 12}
    plt.rc('font', **font)

    plt.clf()
    ax = plt.subplot()
    y_array=y_true+y_pred
    cf_matrix = confusion_matrix(y_true, y_pred,labels=np.unique(y_array))

    df_cm2 = pd.DataFrame(cf_matrix,)
    plt.figure(figsize=(20, 20))
    ax=sn.heatmap(df_cm2, annot=True)
    #ax.set_title(f'{model_name}',fontsize=48)
    plt.xlabel('Predicted',fontsize=24.0, fontweight='bold')
    plt.ylabel('Ground Truth',fontsize=24.0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{type}.png'))
    plt.clf()
    plt.close('all')


def save_to_txt(test_acc, test_top5_acc, classification_report_save_path, y_true, y_pred):
    """
    save result to txt file
    """
    with open(classification_report_save_path, 'a') as f:
        f.write(f'model test acc: {test_acc:.3f}, test top5 acc: {test_top5_acc:.3f}\n')
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write('\n')
def save_length_to_excel(pt_file,num2class):
    """
    save each  class's length to excel file
    """
    length_dict = pt_file['real_length_dict']
    df = pd.DataFrame(
        {'class id': num2class.keys(), 'class name': num2class.values(),'real max length': length_dict.values()})
    df.to_excel(os.path.join('all_max_length_47.xlsx'))
def inference():
    """
    run inference and calculate average time
    """
    start_time = time.time()

    is_save_summary = True
    isgrayscale = False
    issegmented = False
    is_save_confusion_matrix = False
    input_num = 1  ####need setup this value,ex:1,3
    use_length_detection = False
    is_save_length_to_excel=False
    #init
    load_model_time_list = []
    load_image_time_list = []
    inference_time_list = []
    ld_time_list = []
    # init summary
    top1_dic, top5_dic = {}, {}
    ld_top1_dic, ld_top5_dic = {}, {}
    top1, top3, top5 = 0, 0, 0
    if use_length_detection:
        ld_top1, ld_top3, ld_top5 = 0, 0, 0
    total = 0
    top_5_num_correct = 0
    top_1_num_correct = 0

    # save result list init
    pd_img_name = []
    pd_class_idx = []

    pd_class_name = []
    pd_probs_list = []
    pd_preds_list = []
    pd_ld_probs_list = []
    pd_ld_preds_list = []
    pd_total_class_img_num_dict = {}
    pd_class_name_list = {}
    ld_pd_class_name_list = {}

    y_pred = []
    y_true = []
    ld_y_pred = []

    # =====set model file path =====
    test_folder = "t2i no seg"#'test_Fast_seg'
    base = os.path.join("home","user","ISSL-lab-3D-classification-project-main")
    pretrained_root = os.path.join('records', 'FGVC-HERBS', 'baseline(ImageNet)_baseline_avg-20_max_M58-Total(104)_withoutbg_final_HERBS')
    # pretrained_root = os.path.join('records', 'FGVC-HERBS', 'topex-background-t2i_HERBS')
    # test_image_path = './dataset/topex-printer/topex-printer/test/rgb'
    # test_image_path = os.path.join('dataset', 'topex-printer', 'topex-printer', 'test',test_folder)  # similiar_part_test
    # test_image_path = os.path.join('length_detection/data_2image',test_folder)  # similiar_part_test
    test_image_path = os.path.join('length_detection/data_2image',test_folder)  # similiar_part_test

    parser = argparse.ArgumentParser("Visualize SwinT Large")
    # load model
    tmp_time = time.time()
    model_pt_path = os.path.join(pretrained_root, "save_model", "03011-best.pth")
    pt_file = torch.load(model_pt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    class2num = pt_file['class2num']
    num2class = dict((value, key) for key, value in class2num.items())
    # save length to excel
    if is_save_length_to_excel:
        save_length_to_excel(pt_file,num2class)

    model = load_model(pt_file, model_pt_path)
    load_model_time_list.append(time.time() - tmp_time)

    # '''當初忘記更改104的log檔,故新增,之後就不用了'''
    length_detection_dict = pt_file['real_length_dict']


    # get folder list
    cls_folders = os.listdir(test_image_path)
    cls_folders.sort()

    # count n_samples
    n_samples = count_total_pick_times(input_num, test_image_path, class2num, cls_folders)

    pbar = tqdm.tqdm(total=n_samples, ascii=True)


    '''長度測試'''
    estimate_length_dict = {
        'M58-Encoder側走行機身' : 0.436,
        'M58-FORK拖鏈支撐板' : 1.315,
        'M58-FORK拖鏈支撐板-2': 1.323,
        'M58-Fork機身-1' : 1.352,
        'M58-Fork機身-2' : 1.093,
        'M58-Fork機身連接板' : 1.326,
        'M58-FORK驅動軸心' : 0.561,
        'M58-下方護蓋1(從動側)' : 1.036,
        'M58-下方護蓋1(驅動側)' : 1.181,
        'M58-下封板' : 1.048,
        'M58-加熱器固定板' : 0.274,
        'M58-走行從動輪固定座' : 0.224,
        'M58-走行驅動軸心' : 1.406,
        'M58-車身連接補強鋁擠型' : 1.185,
        'M58-拖鏈封蓋' : 1.319,
        'M58-封板支撐柱' : 0.430,
        'M58-乘載板' : 1.316,
        'M58-乘載板-2' : 1.316,
        'M58-乘載板導條' : 1.280,
        'M58-時規皮帶齒條' : 1.366,
        'M58-從動側FORK機身' : 1.033,
        'M58-從動側外殼' : 1.023,
        'M58-從動側走行機身' : 1.018,
        'M58-集電軌側走行機身' : 0.459,
        'M58-滑軌墊片' : 0.875,
        'M58-煞車電阻盤' : 0.515,
        'M58-機身中牽-3' : 1.141,
        'M58-驅動側FORK機身' : 1.054,
        'M58-驅動側外殼' : 1.121
    } 


    for ci, cf in enumerate(cls_folders):
        # get class name from folder multiple_image_run_evaluation.py
        class_name = cf.split('.iam')[0].split('.ipt')[0]
        pd_class_name_list[class2num[class_name]] = class_name
        ld_pd_class_name_list[class2num[class_name]] = class_name
        top1_dic[class2num[class_name]] = top1_dic.get(class2num[class_name], 0)
        ld_top1_dic[class2num[class_name]] = ld_top1_dic.get(class2num[class_name], 0)
        top5_dic[class2num[class_name]] = top5_dic.get(class2num[class_name], 0)
        ld_top5_dic[class2num[class_name]] = ld_top5_dic.get(class2num[class_name], 0)
        files = os.listdir(os.path.join(test_image_path, cf))
        files.sort()
        img_paths = []
        update_n = 0
        selected_paths = [] 
        
        if input_num >= 1:
            # set pick time
            pick_times = len(files) // input_num
            if len(files) % input_num != 0:
                pick_times += 1
            tmp = set()
            for i in range(pick_times):
                # add select path into set
                for path_list in selected_paths:
                    for j in path_list:
                        tmp.add(j)
                remaining_paths = list(set(files) - tmp)
                remaining_paths.sort()
                selected_paths.append(choose_random_paths(files, remaining_paths, input_num))

        for img_path in selected_paths:
            tmp_probs_list = []
            tmp_preds_list = []
            pd_total_class_img_num_dict[class2num[class_name]] = 1 + pd_total_class_img_num_dict.get(
                class2num[class_name], 0)

            for idx, img_name in enumerate(img_path):
                tmp_time = time.time()

                # record predict result
                pd_img_name.append(img_name)
                pd_class_idx.append(class2num[class_name])
                pd_class_name.append(class_name)

                img_path = os.path.join(test_image_path, cf, img_name)
                img_paths.append(img_path)
                img_loader = test_ImgLoader(img_size=pt_file['img_size'],
                                            isgrayscale=isgrayscale)
                img = img_loader.load(img_path)
                imgs = img.unsqueeze(0).cuda()  # add batch size dimension
                load_image_time_list.append(time.time() - tmp_time)
                with torch.no_grad():
                    tmp_time = time.time()
                    imgs = imgs.cuda()
                    outs = model.forward(imgs)

                    if 'HERBS' == pt_file['model_name']:
                        sum_outs = sum_all_out(outs, sum_type="softmax")  # softmax
                        probs, preds = torch.sort(sum_outs, dim=-1, descending=True)
                    else:
                        pred = torch.nn.functional.softmax(outs, -1)
                        probs, preds = pred.topk(len(class2num), dim=1)

                    inference_time_list.append(time.time() - tmp_time)
                    if is_save_confusion_matrix:
                        y_pred.append(preds[0][0].cpu())  # Save Prediction
                        label = class2num[class_name]  # .numpy()
                        y_true.append(label)
                tmp_preds_list.append(preds[0])
                tmp_probs_list.append(probs[0])

            with torch.no_grad():
                if input_num > 1:
                    preds, probs = avg_result(tmp_probs_list, tmp_preds_list)

                if is_save_summary and input_num ==1:
                    pd_probs_list.append(probs[0].cpu())
                    pd_preds_list.append(preds[0].cpu())

                if use_length_detection:
                    tmp_time = time.time()
                    # new_preds, new_probs = length_detection(pt_file['real_length_dict'], class_name, preds, probs, num2class)
                    new_preds, new_probs = length_detection(length_detection_dict, class_name, preds, probs,num2class)
                    # new_preds, new_probs = length_detection_withReal(length_detection_dict, class_name, preds, probs,num2class,estimate_length_dict)
                    ld_time_list.append(time.time() - tmp_time)
                    if is_save_confusion_matrix:
                        ld_y_pred.append(new_preds[0][0].cpu())
                    if is_save_summary and input_num == 1:
                        pd_ld_probs_list.append(new_probs[0].cpu())
                        pd_ld_preds_list.append(new_preds[0].cpu())


            # length detection count correct
            if use_length_detection:
                if class2num[class_name] in new_preds[0][:1]:
                    ld_top1 += 1
                    if is_save_summary and input_num == 1:
                        ld_top1_dic[class2num[class_name]] = 1 + ld_top1_dic.get(class2num[class_name], 0)
                if class2num[class_name] in new_preds[0][:3]:
                    ld_top3 += 1
                if class2num[class_name] in new_preds[0][:5]:
                    ld_top5 += 1
                    if is_save_summary and input_num == 1:
                        ld_top5_dic[class2num[class_name]] = 1 + ld_top5_dic.get(class2num[class_name], 0)

            if class2num[class_name] in preds[0][:1]:
                top1 += 1
                top_1_num_correct += 1
                if is_save_summary and input_num == 1:
                    top1_dic[class2num[class_name]] = 1 + top1_dic.get(class2num[class_name], 0)
            if class2num[class_name] in preds[0][:3]:
                top3 += 1
            if class2num[class_name] in preds[0][:5]:
                top5 += 1
                top_5_num_correct += 1
                if is_save_summary and input_num == 1:
                    top5_dic[class2num[class_name]] = 1 + top5_dic.get(class2num[class_name], 0)

            update_n += 1
            total += update_n
            top1_acc = round(top1 / total * 100, 3)
            top5_acc = round(top5 / total * 100, 3)
            if use_length_detection:
                ld_top1_acc = round(ld_top1 / total * 100, 3)
                ld_top5_acc = round(ld_top5 / total * 100, 3)

            msg = "top1: {}%, top5: {}%".format(top1_acc, top5_acc)
            pbar.set_description(msg)
            pbar.update(update_n)
            update_n = 0
    pbar.close()

    if use_length_detection:
        print(f'use length detection top1: {ld_top1_acc}%,top5: {ld_top5_acc}%')
    if is_save_confusion_matrix:
        save_confusion_matrix(y_true, y_pred, class2num, pretrained_root, 'before LD',
                              f'{pt_file["model_name"]}(proposed)_without_LD')
        save_confusion_matrix(y_true, ld_y_pred, class2num, pretrained_root, 'after LD',
                              f'{pt_file["model_name"]}(proposed)_with_LD')
    if is_save_summary and input_num == 1:
        df = pd.DataFrame(
            {'class id': pd_class_idx, 'class name': pd_class_name, 'img name': pd_img_name, 'preds': pd_preds_list,
             'probs': pd_probs_list})
        df.to_excel(os.path.join(pretrained_root, f'output_{test_folder}.xlsx'))

        if use_length_detection:
            df = pd.DataFrame(
                {'class id': pd_class_idx, 'class name': pd_class_name, 'img name': pd_img_name, 'preds': pd_ld_preds_list,
                'probs': pd_ld_probs_list})
            df.to_excel(os.path.join(pretrained_root, f'output_ld_{test_folder}.xlsx'))

        # top1_acc_list = [str(round(a / b, 3)) + f'({a})' for a, b in
        #                  zip(top1_dic.values(), pd_total_class_img_num_dict.values())]
        # top5_acc_list = [str(round(a / b, 3)) + f'({a})' for a, b in
        #                  zip(top5_dic.values(), pd_total_class_img_num_dict.values())]

        top1_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in zip(top1_dic.values(), pd_total_class_img_num_dict.values())]
        top5_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in zip(top5_dic.values(), pd_total_class_img_num_dict.values())]

        df = pd.DataFrame(
            {'class id': top1_dic.keys(), 'class name': pd_class_name_list.values(),
             'total num img': pd_total_class_img_num_dict.values(), 'top1_correct': top1_dic.values(),
             'top1_acc': top1_acc_list, 'top5_correct': top5_dic.values(), 'top5_acc': top5_acc_list, })
        df.to_excel(os.path.join(pretrained_root, f'ori_{test_folder}.xlsx'))
        top1_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in zip(ld_top1_dic.values(), pd_total_class_img_num_dict.values())]
        top5_acc_list = [str(round((a / b) * 100, 2)) + '%' for a, b in zip(ld_top5_dic.values(), pd_total_class_img_num_dict.values())]

        if use_length_detection:
            df = pd.DataFrame(
                {'class id': ld_top1_dic.keys(), 'class name': pd_class_name_list.values(),
                'total num img': pd_total_class_img_num_dict.values(), 'top1_correct': ld_top1_dic.values(),
                'top1_acc': top1_acc_list, 'top5_correct': ld_top5_dic.values(), 'top5_acc': top5_acc_list, })
            df.to_excel(os.path.join(pretrained_root, f'ld_{test_folder}.xlsx'))
    end_time = time.time()
    print(f'avg time={(end_time - start_time) / n_samples}')

def load_model(pt_file,model_pt_path):
    """
    load model and pretrained weight
    """
    if 'HERBS' == pt_file['model_name']:
        model = build_model(pretrainewd_path=model_pt_path,
                            img_size=pt_file['img_size'],
                            fpn_size=pt_file['fpn_size'],
                            num_classes=pt_file['num_class'],
                            num_selects=pt_file['num_selects'])
    else:
        batch_size = 32
        learning_rate = 0.001
        model = clf_model(len(pt_file['class2num']), learning_rate, batch_size, pt_file['model_name']).to('cuda')
        model.load_state_dict(pt_file['model_state_dict'])
    model.cuda()
    model.eval()
    return model

if __name__ == "__main__":
    inference()
