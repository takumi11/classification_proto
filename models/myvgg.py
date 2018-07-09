# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class Block(chainer.Chain):
    def __init__(self, out_channels, ksize=3, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=None, out_channels=out_channels,
                ksize=ksize, stride=1, pad=pad)
            self.bn_conv = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn_conv(h)
        h = F.relu(h)
        return h


class MyVGG(chainer.Chain):

    def __init__(self, class_num):
        super(MyVGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = Block(24)
            self.conv2_1 = Block(24)
            self.conv3_1 = Block(48)
            self.conv3_2 = Block(48)
            self.conv4_1 = Block(96)
            self.conv4_2 = Block(96)
            self.conv5_1 = Block(128)
            self.conv5_2 = Block(128)
            self.fc1 = L.Linear(None, 512)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, 512)
            self.bn_fc2 = L.BatchNormalization(512)
            self.fc3 = L.Linear(None, class_num)

    def __call__(self, x):
        h = self.conv1_1(x)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv2_1(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)

        h = self.fc2(h)
        h = self.bn_fc2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)

        h = self.fc3(h)

        return h
