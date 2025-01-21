import os
from tqdm import tqdm
import torch
from torchvision import transforms
import shutil
from data.dataset import get_class2num
from PIL import Image

def calculate_non_black_pixel_ratio(image_tensor):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    image = transform(image_tensor).to('cuda')
    total_pixels = image.numel()
    non_black_ratio = torch.nonzero(image).size(0) / total_pixels
    return non_black_ratio

def truncate_new_dataset(root_path,save_path,div_num, partial=False):
    if partial:
        print(f'start filter partial train data..., div num={div_num}')
    else:
        print(f'start filter train data..., div num={div_num}')
    #rm old folder
    if os.path.isdir(save_path) and partial == False:
        print('remove old truncate folder')
        shutil.rmtree(save_path, ignore_errors=True)

    folder_list = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    folder_list.sort()
    class2num = get_class2num(root_path)
    pd_class_id = []
    pd_class_name = []
    pd_num = []

    for folder in tqdm(folder_list):
        # if folder != 'M11-走行從動車輪': continue
        total_percent_sum = 0.0
        img_list = [os.path.join(root_path, folder, name) for name in os.listdir(os.path.join(root_path, folder))]
        total_ratio_list = []
        for img_path in img_list:
            # Read image using torchvision
            # image = read_image(img_path).to('cuda')
            image = Image.open(img_path)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Calculate non-black pixel ratio
            non_black_ratio = calculate_non_black_pixel_ratio(image)
            total_ratio_list.append(non_black_ratio)
            total_percent_sum += non_black_ratio

        zip_dict = dict(map(lambda i, j: (i, j), img_list, total_ratio_list))
        sorted_pairs = sorted(zip(list(zip_dict.values()), list(zip_dict.keys())), reverse=True)
        percent_list, path_list = zip(*sorted_pairs)
        mean_index = percent_list.index(min([k for k in percent_list if k >= (total_percent_sum / len(img_list))]))
        # tqdm.write(f'\n{folder} : {total_percent_sum / len(img_list)},mean index:{mean_index}')
        # tqdm.write(f'max:{max(total_ratio_list)},min:{min(total_ratio_list)}')
        if div_num == 0:
            stop_index = mean_index
        else:
            stop_index = mean_index + (len(percent_list) - 1 - mean_index) // div_num
        # stop_index = (360/2)-1
        pd_class_id.append(class2num[folder])
        pd_class_name.append(folder)
        pd_num.append(stop_index+1)

        # create new dataset
        for idx, i in enumerate(percent_list):
            if idx > stop_index: break
            os.makedirs(os.path.join(save_path, folder), mode=0o777, exist_ok=True)
            shutil.copy(path_list[idx], os.path.join(os.path.join(save_path, folder),f'{idx + 1}_{percent_list[idx]}{os.path.split(path_list[idx])[-1]}'))
        # shutil.copy(os.path.join(root_path, 'log.txt'), os.path.join(os.path.join(save_path, 'log.txt')))
        # df = pd.DataFrame({'class id': pd_class_id, 'class name': pd_class_name, 'index': pd_num, })
        # df.to_excel(os.path.join('qq.xlsx'))

    print('filter train data finish...')