#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os

import chainer
import numpy as np
import cv2

from chainer import Variable
import chainer.function as F
from chainer.dataset import concat_examples
from chainer import cuda
if chainer.cuda.available:
    import cupy as cp


def grad_cam(model, val, gpu, save_dir, backward_label='true_label'):

    xp = cp if gpu >= 0 else np

    # batchサイズ(今回は1枚ずつ行うので1)
    cam_iter = chainer.iterators.SerialIterator(
        val, 1, repeat=False, shuffle=False)
    count = 0

    while True:
        # brakeするまで続ける
        X_test_batch = cam_iter.next()
        # テストデータをデータとラベルに分ける
        X_test, y_test = concat_examples(X_test_batch, gpu)

        with chainer.using_config("train", False):
            pred = model.predictor(X_test)

        probs = F.softmax(pred).data[0]

        if gpu >= 0:
            probs = chainer.cuda.to_cpu(probs)

        top1 = np.argsort(probs)[::-1][0]

        pred.zerograd()
        pred.grad = xp.zero([1, 8], dtype=np.float32)

        if backward_label == 'true_label':
            backward_label = y_test[0]
        else:
            backward_label = top1

        pred.grad[0, backward_label] = 1
        pred.backward(True)

        feature = model.predictor.cam.data[0]
        grad = model.predictor.cam.grad[0]

        cam = xp.ones(feature.shape[1:], dtype=xp.float32)

        weights = grad.mean((1, 2))*1000
        for i, w in enumerate(weights):
            cam += feature[i] * w

        if gpu >= 0:
            cam = chainer.cuda.to_cpu(cam)
            X_test = chainer.cuda.to_cpu(X_test)

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        heatmap = cam / np.mat(cam)
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    ###########################################################################

        image = X_test[0, ::-1, :, :].transpose(1, 2, 0)
        image -= np.min(image)
        image = np.minimum(image, 255)
        cam_img = np.float32(image)
        cam_img = np.float32(heatmap) + np.float32(image)
        cam_img = 255 * cam_img / np.max(cam_img)

        save_img_dir = save_dir + '/visualize'
        if not os.exists(save_img_dir):
            os.mkdir(save_img_dir)
        save_name = save_img_dir + '/cam_img_{}.png'.format(count)
        cv2.imwrite(save_name, cam_img)
        count += 1

        if cam_iter.is_new_epoch:
            cam_iter.epoch = 0
            cam_iter.current_position = 0
            cam_iter.is_new_epoch = False
            cam_iter._pushed_position = None
            break
