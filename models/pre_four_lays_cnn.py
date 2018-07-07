#! /usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class PreFOURLAYSCNN(chainer.Chain):

    def __init__(self, n_out):
        super(PreFOURLAYSCNN, self).__init__()
        with self.init_scope():

            self.conv1 = L.Convolution2D(None, 48, ksize=5, stride=2)
            self.conv2 = L.Convolution2D(None, 48, ksize=4)
            self.conv3 = L.Convolution2D(None, 96, ksize=5, pad=2)
            self.conv4 = L.Convolution2D(None, 192, ksize=5, pad=1)

            self.fc1 = L.Linear(None, 1024)
            self.fc2 = L.Linear(None, n_out)

    def __call__(self, x):

        h = self.conv1(x)

        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = self.conv4(h)
        h = F.relu(h)
        self.cam = h
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.local_response_normalization(h)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)

        h = self.fc2(h)

        return h
