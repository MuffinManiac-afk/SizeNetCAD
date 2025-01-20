# SizeNetCAD : Industrial Component Classification Framework using CAD Rendering and Length-Based Validation on Hololens 2
## Abstract
The machine manufacturing industry increasingly relies on vision-based deep learning models for automated component sorting, requiring large training datasets that are time-intensive to collect. CAD-rendered images are frequently employed as substitutes for real component images despite their distribution gaps that impact classification accuracy. Although photorealistic rendering techniques address these gaps, they often require fine-tuning with real images and are insufficient for distinguishing visually similar components with size variations. This paper proposes SizeNetCAD, a two-stage framework for classifying RGB-D HoloLens 2 industrial component images. The framework incorporates classifier training on CAD-rendered images without surface textures, followed by depth-based length estimation to address size variation challenges across a diverse range of components' shapes. The proposed approach achieves a mean absolute error (MAE) of less than 1 cm for small components and under 5 cm for large components. This results in a 15% improvement in classification accuracy on the Holo RGB-D dataset and a 4% improvement on public datasets compared to predictions without length estimation. Furthermore, the method demonstrates competitive accuracy relative to models trained on photorealistic images. SizeNetCAD facilitates scalable dataset generation and practical deployment by eliminating the need for real images.

## Contribution :
* We proposed a novel SizeNetCAD framework to classify RGB-D industrial component images captured using HoloLens 2, trained using cost-effective CAD-rendered images.
* We conducted extensive length estimation experiments using various component shapes, demonstrating a stable approach with an MAE of less than 1 cm.
* We proposed Length-based classification refinement that demonstrates significant accuracy improvement by 15\% in fine-grained components where inherent features are critical.

## Method
The 3D-Arch pipeline comprises four main modules: (A) data preprocessing, (B) deep learning classification model, (C) length estimation, and (D) length-based classification refinement. The data preprocessing module includes two sub-modules: (1) training data generation, which converts 3D models into rendered 2D images $(x_i)$. and (2) testing data collection and preprocessing, which processes images captured using the HoloLens $(\bar{xa_i}$ and $\bar{xb_i})$. The output of sub-module (2) consists of segmented images $(\bar{xaseg_i}$ and $\bar{xbseg_i})$ . The segmented image $(\bar{xaseg_i})$ is passed to module (B) for inference, generating a prediction list $(y_\alpha)$ , while $(\bar{xbseg_i})$ is used in module (C) to localize the object in the depth image $depth_i$. Finally, the prediction list $(y_\alpha)$is refined in module (D) using the estimated length $(\bar{m_i})$from module (C) to filter predictions based on their lengths$(m_i)$. 


## Dataset Preparation
### Topex-Printer 
Please refer to official topex-printer dataset in [Hugging face](https://huggingface.co/datasets/ritterdennis/topex-printer). If you want to recreate the result from this dataset, please install [ZoeDepth](https://github.com/isl-org/ZoeDepth) package. 
### HoloRGB-D
Private dataset can't be published due to company policy. However, if you happens to have Hololens 2 and CAD model. You can use this package to do classification. Please save the all callibration in excel file, and separate the capture depth image and RGB images to different folders.
## Getting Started
Please install all requirement using conda install <br />  
```conda install -e environment.yaml```<br /> 
All configuration can be found in ```config/configs.yaml```.
### Requirements

### Pretrained FastSAM
Please refer to [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) for installation. There are several pre-trained models, please use 
### HERBS Backbone

## Acknowledgment
* [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
* [HERBs](https://arxiv.org/abs/2303.06442)
* [Topex Printer](https://huggingface.co/datasets/ritterdennis/topex-printer)
  
## Cite
