# -*- coding:utf-8 -*-
__author__ = 'Leo.Z'

import os
import sys
import os.path
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
# 对所有图片生成path-label map.txt

# 实现MyDatasets类
class MyDatasets(datasets.CIFAR10):

    def __init__(self, root, train=True,type = 2, transform=None, target_transform=None,
                 download=False):

        super(MyDatasets, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append([entry['data'][i] for i in range(len(entry['data'])) if entry['labels'][i] == type])
                # self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend([i for i in entry['labels'] if i == type])
                else:
                    self.targets.extend([i for i in entry['fine_labels'] if i == type])
                    # self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()


def quantize_func(num_colors,batch):
    """Takes as input a float tensor with values in the 0 - 1 range and
    outputs a long tensor with integer values corresponding to each
    quantization bin.

    Parameters
    ----------
    batch : torch.Tensor
        Values in 0 - 1 range.
    """
    if num_colors == 2:
        return (batch > 0.5).long()
    else:
        return (batch * (num_colors - 1)).long()


if __name__ == '__main__':
    # 生成map.txt
    # generate_map('train/')
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: quantize_func(256,x))
    ])

    train_loader = DataLoader(MyDatasets('../cifar10_data', train=True, download=False,transform=all_transforms), batch_size=64, shuffle=True)
    print(train_loader)
    for idx, (img, label) in enumerate(train_loader):
        print(img[0][0])
        print(label.shape)