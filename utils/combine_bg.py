import copy
import random
import os,shutil
from tqdm import tqdm
import concurrent.futures
from PIL import Image
from PIL import Image, ImageOps

def remove_component_bg(img,color=None):
    """
    convert background into transparent(RGBA)
    """
    img = img.convert('RGBA')
    image = copy.deepcopy(img)
    newImage = []
    if not color:
        for item in image.getdata():
            if all(0 <= i <= 21 for i in item[:3]):
                newImage.append((0, 0, 0, 0))
            else:
                newImage.append(item)
    else:
        for item in image.getdata():
            if item[:3] == (0, 0, 0):
                newImage.append((0, 0, 0, 0))
            else:
                newImage.append(item)
    image.putdata(newImage)
    return image

def resize_image(img, expected_size):
    """
    Resize image
    """
    from PIL import ImageOps
    img.thumbnail((expected_size[0], expected_size[1]))
    return ImageOps.expand(img)
def combine_bg_fg(image1, save_path,bg_path):
    """
    Combine background and foreground into single image
    """
    img_size=(1920, 1080)
    bg_path = os.path.join(bg_path)
    bg_img_list = os.listdir(bg_path)
    bg_img_path = random.choice(bg_img_list)
    bg = Image.open(os.path.join(bg_path,bg_img_path))
    bg = bg.convert('RGBA')
    bg = bg.resize(img_size)
    scale=1
    image1 = add_padding_to_image(image1, img_size[0], img_size[1], padding_color=(0, 0, 0,0))
    image1=resize_image(image1, img_size)
    bg.paste(image1, (0,0), image1)
    bg.convert('RGB').save(save_path)

def add_padding_to_image(image, target_width, target_height, padding_color=(0, 0, 0,0)):
    current_width, current_height = image.size
    
    delta_width = max(0, target_width - current_width)
    delta_height = max(0, target_height - current_height)
    padding = (
        delta_width // 2,  # Left
        delta_height // 2,  # Top
        delta_width - (delta_width // 2),  # Right
        delta_height - (delta_height // 2)  # Bottom
    )
    
    padded_image = ImageOps.expand(image, border=padding, fill=padding_color)
    return padded_image

def process_image(img_path, save_path,bg_path):
    """
    execute combine process
    """
    with Image.open(img_path) as img:
        rgba_img = remove_component_bg(img)
        combine_bg_fg(rgba_img, save_path, bg_path)

def combine_bg(args,base_train_path,save_path,x_train):
    if os.path.isdir(save_path):
        print('remove old combine folder')
        shutil.rmtree(save_path, ignore_errors=True)

    num_cores = os.cpu_count()-2

    total_img_num = 0

    pbar = tqdm(total=total_img_num, ascii=True)
    for img_path in x_train:
            total_img_num += 1
            base_path = save_path
            os.makedirs(name=os.path.join(base_path), mode=0o777, exist_ok=True)
            folder_path = img_path.split('/')[-1].split('\\')[1]
            img_save_path=img_path.split('/')[-1].split('\\')[-1]
            os.makedirs(name=os.path.join(base_path, folder_path), mode=0o777, exist_ok=True)
            output_img_path = os.path.join(base_path, folder_path,img_save_path)
            process_image(img_path, output_img_path,args.bg_path)
            pbar.update(1)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
    #     futures = []
    #     for img_path in x_train:
    #         total_img_num += 1
    #         base_path = save_path
    #         os.makedirs(name=os.path.join(base_path), mode=0o777, exist_ok=True)
    #         folder_path = img_path.split('/')[-1].split('\\')[1]
    #         img_save_path=img_path.split('/')[-1].split('\\')[-1]
    #         os.makedirs(name=os.path.join(base_path, folder_path), mode=0o777, exist_ok=True)
    #         output_img_path = os.path.join(base_path, folder_path,img_save_path)
    #         # process_image(img_path, output_img_path)
    #         futures.append(executor.submit(process_image, img_path, output_img_path,args.bg_path))
    #     pbar = tqdm(total=total_img_num, ascii=True)
    #     for future in concurrent.futures.as_completed(futures):
    #         pbar.update(1)
    # pbar.close()


    # for future in futures:
    #     future.result()

    shutil.copy(os.path.join(base_train_path, 'log.txt'), os.path.join(os.path.join(save_path, 'log.txt')))

if __name__ == "__main__":
    pass


