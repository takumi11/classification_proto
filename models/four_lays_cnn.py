#! /usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class FOURLAYSCNN(chainer.Chain):

    def __init__(self, n_out):
        super(FOURLAYSCNN, self).__init__()
        with self.init_scope():

            self.conv1 = L.Convolution2D(None, 48, ksize=5, stride=2)
            self.conv2 = L.Convolution2D(None, 48, ksize=4)
            self.conv3 = L.Convolution2D(None, 96, ksize=5, pad=2)
            self.conv4 = L.Convolution2D(None, 192, ksize=5, pad=1)

            self.fc1 = L.Linear(None, 1024)
            self.fc2 = L.Linear(None, n_out)

            self.bnorm1 = L.BatchNormalization(48)
            self.bnorm2 = L.BatchNormalization(48)
            self.bnorm3 = L.BatchNormalization(96)
            self.bnorm4 = L.BatchNormalization(192)

    def __call__(self, x):

        h = self.conv1(x)
        h = self.bnorm1(h)

        h = self.conv2(h)
        h = self.bnorm2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv3(h)
        h = self.bnorm3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv4(h)
        h = self.bnorm4(h)
        h = F.relu(h)
        self.cam = h
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.fc1(h)
        h = self.fc2(h)

        return h

