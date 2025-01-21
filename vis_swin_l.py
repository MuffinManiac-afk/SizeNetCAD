import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os,time
import argparse
import matplotlib
import matplotlib.pyplot as plt
from vis_utils import vis_ImgLoader, get_cdict
from tqdm import tqdm
global module_id_mapper
global features
global grads
from models.classification_model import clf_model
from utils.gradcam import GradCam
from torchvision import transforms
def forward_hook(module: nn.Module, inp_hs, out_hs):
    global features, module_id_mapper
    layer_id = len(features) + 1
    module_id_mapper[module] = layer_id
    features[layer_id] = {}
    features[layer_id]["in"] = inp_hs
    features[layer_id]["out"] = out_hs
    # print('forward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_hs.size()))

def backward_hook(module: nn.Module, inp_grad, out_grad):
    global grads, module_id_mapper
    layer_id = module_id_mapper[module]
    grads[layer_id] = {}
    grads[layer_id]["in"] = inp_grad
    grads[layer_id]["out"] = out_grad
    # print('backward_hook, layer_id:{}, hs_size:{}'.format(layer_id, out_grad[0].size()))


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
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() if
                           k in model.state_dict() and 'head' not in k}  # ('patch' in k or 'layer' in k or 'norm' in k)}
        model.load_state_dict(pretrained_dict, strict=False)
    
    model.eval()

    ### hook original layer1~4
    model.backbone.layers[0].register_forward_hook(forward_hook)
    model.backbone.layers[0].register_full_backward_hook(backward_hook)
    model.backbone.layers[1].register_forward_hook(forward_hook)
    model.backbone.layers[1].register_full_backward_hook(backward_hook)
    model.backbone.layers[2].register_forward_hook(forward_hook)
    model.backbone.layers[2].register_full_backward_hook(backward_hook)
    model.backbone.layers[3].register_forward_hook(forward_hook)
    model.backbone.layers[3].register_full_backward_hook(backward_hook)
    ### hook original FPN layer1~4
    model.fpn_down.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_down.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_down.Proj_layer4.register_full_backward_hook(backward_hook)
    ### hook original FPN_UP layer1~4
    model.fpn_up.Proj_layer1.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer1.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer2.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer2.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer3.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer3.register_full_backward_hook(backward_hook)
    model.fpn_up.Proj_layer4.register_forward_hook(forward_hook)
    model.fpn_up.Proj_layer4.register_full_backward_hook(backward_hook)

    return model

def cal_backward(args, out, sum_type: str = "softmax"):
    assert sum_type in ["none", "softmax"]

    target_layer_names = ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 'comb_outs']

    sum_out = None
    for name in target_layer_names:

        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)

        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error

    with torch.no_grad():
        if args.use_label:
            print("use label as target class")
            pred_score = torch.softmax(sum_out, dim=-1)[0][args.label]
            backward_cls = args.label
        else:
            pred_score, pred_cls = torch.max(torch.softmax(sum_out, dim=-1), dim=-1)
            pred_score = pred_score[0]
            pred_cls = pred_cls[0]
            backward_cls = pred_cls

    #print(sum_out.size())
    #print("pred: {}, gt: {}, score:{}".format(backward_cls, args.label, pred_score))
    sum_out[0, backward_cls].backward()

@torch.no_grad()
def get_grad_cam_weights(grads):
    weights = {}
    for grad_name in grads:
        _grad = grads[grad_name]['out'][0][0]
        L, C = _grad.size()
        H = W = int(L ** 0.5)
        _grad = _grad.view(H, W, C).permute(2, 0, 1)
        C, H, W = _grad.size()
        weights[grad_name] = _grad.mean(1).mean(1)
        #print(weights[grad_name].max())

    return weights

@torch.no_grad()
def plot_grad_cam(features, weights):
    act_maps = {}
    for name in features:
        hs = features[name]['out'][0]
        L, C = hs.size()
        H = W = int(L ** 0.5)
        hs = hs.view(H, W, C).permute(2, 0, 1)
        C, H, W = hs.size()
        w = weights[name]
        w = w.view(-1, 1, 1).repeat(1, H, W)
        weighted_hs = F.relu(w * hs)
        a_map = weighted_hs
        a_map = a_map.sum(0)
        # a_map /= abs(a_map).max()
        act_maps[name] = a_map
    return act_maps


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
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out  # note that use '+=' would cause inplace error
    return sum_out/len(target_layer_names)

def visualize_image():
    """
    output heatmap into folder
    """
    isgrayscale = False
    is_show_top_5_prediction = False
    save_folder_name = 'heatmap'
    start_time = time.time()
    # ===== get setting =====
    pretrained_root = os.path.join('records', 'FGVC-HERBS', 'baseline(ImageNet)_baseline_avg-20_max_M58-Total(104)_withoutBG_HERBS')
    test_image_path = os.path.join('dataset', 'M58', 'report_test_104')  # test_0301

    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument("-img", "--image", type=str, default=f'{test_image_path}', )
    parser.add_argument("-sn", "--save_name", type=str, default=f'1', )
    parser.add_argument("-lb", "--label", type=int)
    parser.add_argument("-usl", "--use_label", default=False, type=bool)
    parser.add_argument("-sum_t", "--sum_features_type", default="softmax", type=str)
    args = parser.parse_args()

    # get folder list
    folder_list = os.listdir(test_image_path)

    model_pt_path = os.path.join(pretrained_root, "save_model", "best.pth")
    pt_file = torch.load(model_pt_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    class2num = pt_file['class2num']
    # ===== build model =====
    if 'HERBS' == pt_file['model_name']:
        model = build_model(pretrainewd_path=model_pt_path,
                            img_size=pt_file['img_size'],
                            fpn_size=pt_file['fpn_size'],
                            num_classes=pt_file['num_class'],
                            num_selects=pt_file['num_selects'])
    else:
        batch_size = 32
        learning_rate = 0.001
        model = clf_model(len(class2num), learning_rate, batch_size, pt_file['model_name']).to('cuda')
        model.load_state_dict(pt_file['model_state_dict'])
    model.cuda()
    total_time = 0.0
    class2num = pt_file['class2num']
    n_img = 0
    n_samples = 0
    update_n = 0
    #calculate total image
    for ci, cf in enumerate(folder_list):
        n_samples += len(os.listdir(os.path.join(test_image_path, cf)))

    pbar = tqdm(total=n_samples, ascii=True)

    load_model_time = time.time() - start_time
    # ===== load image =====
    for i, folder in enumerate(folder_list):
        start_time = time.time()
        img_list = os.listdir(os.path.join(test_image_path, f'{folder}'))

        for k, image in enumerate(img_list):
            update_n += 1
            global module_id_mapper, features, grads
            module_id_mapper, features, grads = {}, {}, {}
            n_img += 1
            if k != 0:
                start_time = time.time()
            tmp = time.time()
            img_loader = vis_ImgLoader(img_size=pt_file['img_size'], isgrayscale=isgrayscale)
            img, ori_img = img_loader.load(test_image_path + f'\\{folder}\\{image}')
            # ===== forward and backward =====
            img = img.unsqueeze(0).cuda()  # add batch size dimension
            load_image_time = time.time() - tmp
            tmp = time.time()
            out = model.forward(img)
            inference_time = time.time() - tmp

            tmp = time.time()
            if 'HERBS' == pt_file['model_name']:
                sum_outs = sum_all_out(out, sum_type="softmax")
                probs, preds = torch.sort(sum_outs, dim=-1, descending=True)
            else:
                pred = torch.nn.functional.softmax(out, -1)
                probs, preds = pred.topk(len(class2num), dim=1)
            postprocessing_time = time.time() - tmp
            tmp = time.time() - start_time
            total_time += tmp
            if is_show_top_5_prediction:
                print(
                    f'load model:{load_model_time},load img:{load_image_time},inference:{inference_time},postprocessing:{postprocessing_time}')
                print(f'target:{class2num[folder]},top-5 class:{preds[0][:5]}, top-5 probs:{probs[0][:5]}')
                print(f'total inference time:{tmp}')

            if True:  # class2num[folder] not in preds[0][:5]:
                if 'HERBS' == pt_file['model_name']:
                    cal_backward(args, out, sum_type="softmax")

                    # ===== check result =====
                    grad_weights = get_grad_cam_weights(grads)
                    act_maps = plot_grad_cam(features, grad_weights)

                    # ===== show =====
                    sum_act = None
                    resize = torchvision.transforms.Resize((pt_file['img_size'], pt_file['img_size']))
                    for name in act_maps:
                        layer_name = "layer: {}".format(name)
                        _act = act_maps[name]
                        _act /= _act.max()
                        r_act = resize(_act.unsqueeze(0))
                        act_m = _act.cpu().numpy() * 255
                        act_m = act_m.astype(np.uint8)
                        act_m = cv2.resize(act_m, (pt_file['img_size'], pt_file['img_size']))
                        if sum_act is None:
                            sum_act = r_act
                        else:
                            sum_act *= r_act

                    sum_act /= sum_act.max()
                    sum_act = torchvision.transforms.functional.adjust_gamma(sum_act, 1.0)
                    sum_act = sum_act.cpu().numpy()[0]

                    os.makedirs(name=pretrained_root + f'\\{save_folder_name}\\{folder}', mode=0o777, exist_ok=True)

                    plt.cla()
                    cdict = get_cdict()
                    cmap = matplotlib.colors.LinearSegmentedColormap("jet_revice", cdict)
                    plt.imshow(ori_img[:, :, ::] / 255)
                    plt.imshow(sum_act, alpha=0.5, cmap=cmap)
                    plt.axis('off')
                    plt.savefig(
                        pretrained_root + f'\\{save_folder_name}\\{folder}\\{class2num[folder]}_{preds[0][0]}_{preds[0][1]}_{preds[0][2]}_{preds[0][3]}_{preds[0][4]}_{image}',
                        bbox_inches='tight', pad_inches=0.0, transparent=True)
                    # plt.show()
                    plt.clf()
                    plt.close('all')
                else:
                    grad_cam = GradCam(model, pt_file['model_name'])
                    if pt_file['model_name'] == 'vgg16':
                        target_layer = 30
                    elif pt_file['model_name'] == 'resnet18':
                        target_layer = 1
                    elif pt_file['model_name'] == 'convnext':
                        target_layer = 7

                    feature_image = grad_cam(img, target_layer).squeeze(dim=0)
                    feature_image = transforms.ToPILImage()(feature_image)
                    os.makedirs(name=pretrained_root + f'\\{save_folder_name}\\{folder}', mode=0o777, exist_ok=True)
                    feature_image.save(
                        pretrained_root + f'\\{save_folder_name}\\{folder}\\{class2num[folder]}_{preds[0][0]}_{preds[0][1]}_{preds[0][2]}_{preds[0][3]}_{preds[0][4]}_{image}')
                pbar.update(update_n)
                update_n = 0
if __name__ == "__main__":
    visualize_image()
