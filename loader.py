import torch
from PIL import Image
import os
import pandas as pd
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, num_classes=100, num_train_sample=0, novel_only=False, aug=False,
                 loader=pil_loader):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        # split dataset
        data = data[data['label'] < num_classes]
        base_data = data[data['label'] < 100]
        novel_data = data[data['label'] >= 100]
        
        # sampling from novel classes
        if num_train_sample != 0:
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])

        # whether only return data of novel classes
        if novel_only:
            data = novel_data
        else:
            data = pd.concat([base_data, novel_data])

        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)
        
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)