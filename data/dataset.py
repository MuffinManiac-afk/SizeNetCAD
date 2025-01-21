import random
import numpy as np
import cv2,os
import torch
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torchvision.utils import save_image
def get_train_image_list(dataset_path,translate_class2num):
    """
    get image list and label from dataset_path
    """
    # get image list
    img_path_list = []
    img_classes_list = []

    classes=[name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    for class_ in classes:
        if not os.path.isdir(os.path.join(dataset_path ,class_)):
            continue
        for img in os.listdir(os.path.join(dataset_path ,class_)):
            if (img[-3:] != "jpg") and (img[-3:] != "png"):
                continue
            img_path = os.path.join(dataset_path, class_ , img)
            img_path_list.append(img_path)
            class_name = class_
            img_classes_list.append(translate_class2num[class_name])

    return img_path_list,img_classes_list
def get_class2num(path):
    """
    get part class2num dict

    Args:
        path : dataset path

    Returns:
        class2num: part class2num dict
    """

    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    model_list.sort()
    class2num = {}
    for idx, item in enumerate(model_list):
        class_name = item
        class2num[class_name] = idx
    return class2num

def get_num2class(path):
    """
    get part class2num dict

    Args:
        path : dataset path

    Returns:
        class2num: part class2num dict
    """

    model_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    model_list.sort()
    num2class = {}
    for idx, item in enumerate(model_list):
        class_name = item
        num2class[idx] = class_name
    return num2class
def build_loader(args,is_train_aug=False,fold_img_label=None,img_size=384,batch_size=8):
    """
    get image list and split train,valid dataset, then create dataloader
    """
    if args.isgrayscale==True:
        print('HERBS dataset is grayscale')
    else:
        print('HERBS dataset is RGB')
    class2num = get_class2num(args.train_root)
    num_classes = len(class2num)

    tmp = list(class2num.keys())[0]
    args.machine_type = tmp.split('-')[0]

    original_img_path_list, original_img_classes_list = get_train_image_list(args.train_root, class2num)
    if fold_img_label == None :
        print("[dataset] class number:", num_classes)
        # #split train valid data
        # print(args.train_root)
        print(len(original_img_path_list))
        x_train, x_valid, y_train, y_valid = train_test_split(original_img_path_list, original_img_classes_list,
                                                              test_size=0.2,
                                                              random_state=32, stratify=original_img_classes_list,
                                                              shuffle=True)
    else:
        print('using_cross_validation')
        print("[dataset] class number:", num_classes)
        print(len(original_img_path_list))
        x_train = [original_img_path_list[i] for i in fold_img_label[0]]
        y_train = [original_img_classes_list[i] for i in fold_img_label[0]]
        x_valid = [original_img_path_list[i] for i in fold_img_label[1]]
        y_valid = [original_img_classes_list[i] for i in fold_img_label[1]]
        print(f'train length:{len(x_train)},valid length:{len(x_valid)}')

    train_set, train_loader = None, None
    if x_train is not None:
        train_set = ImageDataset(istrain=True, img_list=x_train,label_list=y_train, data_size=img_size, return_index=True,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='train_aug',filename='ori')
        if is_train_aug:
            train_aug_set = ImageDataset(istrain=True, img_list=x_train, label_list=y_train, data_size=img_size,
                                     return_index=True,is_train_aug=is_train_aug,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='train_aug',filename='aug')
            train_set = torch.utils.data.ConcatDataset([train_set, train_aug_set])
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size,num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,)

    val_set, val_loader = None, None
    if x_valid is not None:
        val_set = ImageDataset(istrain=False, img_list=x_valid,label_list=y_valid, data_size=img_size, return_index=True,isgrayscale=args.isgrayscale,is_save=args.is_save_img,save_folder='valid',filename='valid')
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=2, shuffle=True, batch_size=batch_size,pin_memory=True,
        persistent_workers=True)

    return train_loader, val_loader

class ImageDataset(torch.utils.data.Dataset):
    """
    custom dataset
    """
    def __init__(self,
                 istrain: bool,
                 img_list,
                 label_list,
                 data_size: int,
                 return_index: bool = False,
                 is_train_aug=False,
                 isgrayscale=False,
                 is_save=False,
                 save_folder=None,
                 filename=None,
                 ):
        """ basic information """
        self.img_list = img_list
        self.label_list=label_list
        self.data_size = data_size
        self.return_index = return_index
        self.isgrayscale=isgrayscale
        self.is_save = is_save
        self.is_train_aug = is_train_aug
        self.save_folder = save_folder
        self.filename = filename
        """ declare data augmentation """
        if is_train_aug:
            self.transforms=A.Compose(
            [
                # A.CenterCrop(1080, 1080),
                A.Resize(self.data_size, self.data_size),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45,border_mode=cv2.BORDER_CONSTANT,value=0,p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=0.5),
                A.GaussianBlur (blur_limit=(3, 7),p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        elif istrain:
            self.transforms=A.Compose(
            [
                # A.CenterCrop(1080, 1080),
                A.Resize(self.data_size, self.data_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        else:
            self.transforms = A.Compose(
                [
                    # A.CenterCrop(1080, 1080),
                    A.Resize(self.data_size, self.data_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

        """ read all data information """
        data_infos=[]
        for i,img_path in enumerate(self.img_list):
            data_infos.append({"path": img_path, "label": label_list[i]})
        self.data_infos = data_infos

    def __len__(self):
        return len(self.data_infos)
    def save_image(self,image,index):
        """
        save image

        Args:
            image : image file
            index : index of image
        """
        save_path=os.path.join(self.save_folder)
        os.makedirs(name=save_path, mode=0o777, exist_ok=True)
        save_image(image, os.path.join(save_path,f'{self.filename}_{index}.jpg'),normalize=True)
    def __getitem__(self, index):
        random.seed(index)

        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # =============================
        # read image by opencv.
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if img.shape[-1] == 4:
            img = img[:,:,:-1] # topic data come with 4 bands dataset thus we remove the fourth band
        # isgrayscale = False
        if self.isgrayscale:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.repeat(img[..., np.newaxis], 3, -1)
        else:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, :, ::-1]  # BGR to RGB.

        img = self.transforms(image=img)['image']
        if self.is_save:
            self.save_image(img, index)
        if self.return_index:
            return index, img, label
        return img, label
