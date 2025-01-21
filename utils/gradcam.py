import cv2,os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable


class GradCam:
    def __init__(self, model,model_type):
        self.model = model
        self.feature = None
        self.gradient = None
        self.model_type=model_type

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x,target_layer=30):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)

        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            for i,k in self.model.named_children():
                for idx,(name, module) in enumerate(k.named_children()):
                    if self.model_type=='vgg16':
                        if name == 'classifier':
                            feature = feature.view(feature.size(0), -1)
                            feature = module(feature)
                        elif name == 'features':
                            for j,layer in enumerate(module.children()):
                                feature = layer(feature)
                                if j==target_layer:
                                    feature.register_hook(self.save_gradient)
                                    self.feature = feature
                        elif name == 'avgpool':
                            feature = module(feature)
                    elif self.model_type=='resnet18':
                        if name == 'fc':
                            feature = feature.view(feature.size(0), -1)
                        if name == 'layer4':
                            for k,layer in enumerate(module.children()):
                                feature = layer(feature)
                                if k==target_layer:
                                    feature.register_hook(self.save_gradient)
                                    self.feature = feature
                        else:
                            feature = module(feature)
                    elif self.model_type == 'convnext':
                        if name == 'features':
                            for j,layer in enumerate(module.children()):
                                feature = layer(feature)
                                if j==target_layer:
                                    feature.register_hook(self.save_gradient)
                                    self.feature = feature
                        elif name == 'avgpool':
                            feature = module(feature)
                        elif name == 'classifier':
                            feature = module(feature)

            classes = torch.sigmoid(feature)
            one_hot, predict_class = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()
            self.model.zero_grad()
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)

            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam =  np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))[:, :, ::-1]+ heat_map
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps

