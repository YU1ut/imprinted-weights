# :construction: Work in Progress :construction:
# imprinted-weights
This is an unofficial pytorch implementation of [Low-Shot Learning with Imprinted Weights](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf). 

## Requirements
- Python 3.5+
- PyTorch 0.4
- torchvision
- progress
- matplotlib

## Major Difference: 
Paper: InceptionV1 + RMSProp

This implementation: ResNet-50 + SGD

## Preparation
Download [CUB_200_2011 Dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).

Unzip and locate it in this directory.

The whole directory should be look like this:
```
imprinted-weights
│   README.md
│   pretrain.py
│   models.py
│   loader.py
│   imprint.py
│   
└───CUB_200_2011
    │   images.txt
    │   image_class_labels.txt
    │   train_test_split.txt
    │
    └───images
        │   001.Black_footed_Albatross
        │   002.Laysan_Albatross
        │   ...
```

## Usage
### Pretrain models
Train the model on the first 100 classes of CUB_200_2011.
```
python pretrain.py
```
Trained models will be saved at `pretrain_checkpoint`.

### Imprint weights
Use 1 novel exemplar from the training split to imprint weights.
```
python imprint.py --resume pretrain_checkpoint/checkpoint.pth.tar --num-sample 1
```
For more details and parameters, please refer to --help option.
All w/o FT results of Table 1 and Table 2 in the paper can be reproduced by this script.

## Results
### w/o FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand-noFT (paper) |0.17 |0.17 |0.17 |0.17 |0.17 |
|Rand-noFT (paper) |0.17 |0.17 |0.17 |0.17 |0.17 |
|Imprinting (paper)|21.26 |28.69 |39.52 |45.77 |49.32
|Imprinting + Aug  (paper) |21.40 |30.03 |39.35 |46.35 |49.80