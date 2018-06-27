# :construction: Work in Progress :construction:
# imprinted-weights
This is an unofficial pytorch implementation of [Low-Shot Learning with Imprinted Weights](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf). 

## Requirements
- Python 3.5+
- PyTorch 0.4
- torchvision
- progress

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
Train the model on the first 100 classes of CUB_200_2011.
```
python pretrain.py
```
Trained model will be saved at `pretrain_checkpoint`.

