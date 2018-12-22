# imprinted-weights
This is an unofficial pytorch implementation of [Low-Shot Learning with Imprinted Weights](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Low-Shot_Learning_With_CVPR_2018_paper.pdf). 

## Requirements
- Python 3.5+
- PyTorch 0.4.1
- torchvision
- pandas (0.23.4)
- progress
- matplotlib
- numpy

## Major Difference
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
│   imprint_ft.py
│   alljoint.py
|
└───utils
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
Use N novel exemplar from the training split to imprint weights.
```
python imprint.py --model pretrain_checkpoint/model_best.pth.tar --num-sample N
```

Trained models will be saved at `imprint_checkpoint`.

For more details and parameters, please refer to --help option.

All w/o FT results of Table 1 and Table 2 in the paper can be reproduced by this script.

### Imprint weights + FT
Apply fine-tuning to the imprinting model.
```
python imprint_ft.py --model pretrain_checkpoint/model_best.pth.tar --num-sample N
```
Trained models will be saved at `imprint_ft_checkpoint`.

All w/ FT results of Table 1 and Table 2 in the paper can be reproduced by this script.

### All class joint
Train model for base and novel classes jointly without a separate low-shot learning phase.
```
python alljoint.py --num-sample N
```
Trained models will be saved at `alljoint_checkpoint`.

## Results
### 200-way top-1 accuracy for novel-class examples in CUB-200-2011
#### w/o FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand-noFT (paper) |0.17 |0.17 |0.17 |0.17 |0.17 |
|Imprinting (paper)|21.26 |28.69 |39.52 |45.77 |49.32|
|Imprinting + Aug (paper) |21.40 |30.03 |39.35 |46.35 |49.80|
|**Rand-noFT**|**0.00** |**0.00** |**0.00** |**0.00** |**0.00** |
|**Imprinting** |**28.77** |**37.61** |**50.20** |**56.31** |**60.58**|
|**Imprinting + Aug** |**28.81** |**38.02** |**49.90** |**56.18** |**60.44**|

#### w/ FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand + FT (paper) |5.25 |13.41 |34.95| 54.33 |65.60|
|Imprinting + FT (paper)|18.67 |30.17| 46.08 |59.39 |68.77|
|AllClassJoint (paper) |3.89 |10.82 |33.00 |50.24 |64.88|
|**Rand + FT**|**2.39** |**13.00** |**36.11** |**56.83** |**69.90** |
|**Imprinting + FT** |**26.14** |**34.81** |**54.40** |**63.41** |**73.41** |
|**AllClassJoint** |**7.34** |**18.09** |**45.67** |**60.58** |**73.41** |



### 200-way top-1 accuracy measured across examples in all classes of CUB-200-2011
#### w/o FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand-noFT (paper) |37.36| 37.36| 37.36| 37.36 |37.36|
|Imprinting (paper)|44.75| 48.21| 52.95| 55.99 |57.47|
|Imprinting + Aug (paper) |44.60| 48.48| 52.78 |56.51| 57.84|
|**Rand-noFT**|**41.39** |**41.39** |**41.39** |**41.39** |**41.39** |
|**Imprinting** |**53.50** |**57.35** |**62.65** |**65.38** |**67.09**|
|**Imprinting + Aug** |**53.40** |**57.47** |**62.56** |**65.21** |**67.26**|



#### w/ FT
| n = | 1| 2 | 5| 10| 20|
|:---|:---:|:---:|:---:|:---:|:---:|
|Rand + FT (paper) |39.26 |43.36| 53.69| 63.17| 68.75|
|Imprinting + FT (paper)|45.81 |50.41 |59.15| 64.65| 68.73|
|AllClassJoint (paper) |38.02 |41.89| 52.24| 61.11| 68.31|
|**Rand + FT**|**41.27** |**46.41** |**57.71** |**67.79** |**74.72** |
|**Imprinting + FT** |**53.02** |**57.82** |**67.21** |**71.73** |**76.25** |
|**AllClassJoint** |**42.73** |**48.43** |**62.20** |**69.95** |**75.89** |

## References
- [1]: H. Qi, M. Brown and D. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.