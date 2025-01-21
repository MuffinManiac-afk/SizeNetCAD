import cv2
import numpy as np
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2

class vis_ImgLoader(object):

    def __init__(self, img_size: int,text=None,isgrayscale=False):
        self.text=text
        self.img_size = img_size
        self.isgrayscale=isgrayscale
        self.transform = A.Compose(
            [
                A.Crop(x_min=604, y_min=186, x_max=1314, y_max=896),
                #A.Crop(x_min=489, y_min=41,x_max=1489,y_max=1041),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
                ])

    def load(self, image_path: str):
        ori_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        assert ori_img.shape[2] == 3, "3(RGB) channels is required."
        img = copy.deepcopy(ori_img)
        if self.isgrayscale:
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            img = np.repeat(img[..., np.newaxis], 3, -1)

        img = img[:, :, ::-1] # convert BGR to RGB
        ori_img = ori_img[:, :, ::-1]
        if self.isgrayscale:
            import time
            s=time.time()
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            ori_img = np.repeat(ori_img[..., np.newaxis], 3, -1)

        img = self.transform(image=img)['image']
        t = A.Compose(
            [
                A.Crop(x_min=489, y_min=41,x_max=1489,y_max=1041),
                A.Resize(self.img_size, self.img_size),
            ])

        ori_img=t(image=ori_img)['image']
        return img, ori_img
class test_ImgLoader(object):
    def __init__(self, img_size: int,text=None,isgrayscale=False):
        self.text=text
        self.img_size = img_size
        self.isgrayscale=isgrayscale
        self.transform = A.Compose(
            [
                # A.Crop(x_min=604, y_min=186, x_max=1314, y_max=896),
                # A.Crop(x_min=1000, y_min=186, x_max=1710, y_max=896),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
                ])

    def load(self, image_path: str):
        ori_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        assert ori_img.shape[2] == 3, "3(RGB) channels is required."
        img = copy.deepcopy(ori_img)
        if self.isgrayscale:
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            img = np.repeat(img[..., np.newaxis], 3, -1)

        img = img[:, :, ::-1] # convert BGR to RGB

        img = self.transform(image=img)['image']
        return img
    
    def transfrom(self):
        img = self.transform(image=img)['image']
        return img
    
class test_ImgLoader_FastSAM(object):
    def __init__(self, img_size: int,text=None,isgrayscale=False):
        self.text=text
        self.img_size = img_size
        self.isgrayscale=isgrayscale
        self.transform = A.Compose(
            [
                # A.Crop(x_min=604, y_min=186, x_max=1314, y_max=896),
                # A.Crop(x_min=1000, y_min=186, x_max=1710, y_max=896), #crop new image left side
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
                ])
        
    def load(self, image_path: str):
        ori_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        assert ori_img.shape[2] == 3, "3(RGB) channels is required."
        img = copy.deepcopy(ori_img)
        if self.isgrayscale:
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            img = np.repeat(img[..., np.newaxis], 3, -1)

        # img = img[:, :, ::-1] # convert BGR to RGB

        # img = self.transform(image=img)['image']
        return img
    
    def apply_transform(self, img):
        img = self.transform(image=img)['image']
        return img


def get_cdict():
    _jet_data = {
              # 'red':   ((0.00, 0, 0),
              #          (0.35, 0, 0),
              #          (0.66, 1, 1),
              #          (0.89, 1, 1),
              #          (1.00, 0.5, 0.5)),
              'red':   ((0.00, 0, 0),
                       (0.35, 0.5, 0.5),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.8, 0.8)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0.3, 0.3),
                       (1.000, 0, 0)),
             # 'blue':  ((0.00, 0.5, 0.5),
             #           (0.11, 1, 1),
             #           (0.34, 1, 1),
             #           (0.65, 0, 0),
             #           (1.00, 0, 0))}
             'blue':  ((0.00, 0.30, 0.30),
                       (0.25, 0.8, 0.8),
                       (0.34, 0.8, 0.8),
                       (0.65, 0, 0),
                       (1.00, 0, 0))
             }
    return _jet_data