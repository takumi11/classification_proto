#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import chainer
from functools import partial
from chainercv import transforms


class PreProcess(chainer.dataset.DatasetMixin):

    def __init__(self, pair, mean):
        self.base = pair
        self.mean = mean

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        image, label = self.base[i]
        image -= self.mean
        image = image / 255.0

        return (image, label)


def transform(data, mean, train=True):

    img, lable = data
    img = img.copy()

    if train:
        img = transforms.random_flip(img, x_random=True)
        img = transforms.random_rotate(img, return_param=False)

    img -= mean

    return img / 255., lable


def compute_mean(dataset):

    print('compute mean image')
    sum_image = 0
    N = len(dataset)
    for i, (image, _) in enumerate(dataset):
        sum_image += image
        sys.stderr.write('{} / {}\r'.format(i, N))
        sys.stderr.flush()
    sys.stderr.write('\n')

    return sum_image / N


def get_dataset(dataset):

    length = len(dataset)
    class_num = len(set(dataset.labels))
    print('class_number: {}'.format(class_num))
    print('')
    split_at = int(length * 0.9)

    train_data, test_data = chainer.datasets.split_dataset_random(
        dataset, split_at, 0)
    mean = compute_mean(train_data)
    print('train_data length: {}'.format(len(train_data)))
    print('test_data length: {}'.format(len(test_data)))

    train = chainer.datasets.TransformDataset(
        train_data, partial(transform, mean=mean, train=True))
    test = chainer.datasets.TransformDataset(
        test_data, partial(transform, mean=mean, train=False))

    return train, test, class_num, mean
