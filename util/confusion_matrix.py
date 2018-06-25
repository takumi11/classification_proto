#! /usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer.dataset import concat_examples

import numpy as np
from sklearn.metrics import confusion_matrix


class Confusion_Matrix():
    def __init__(self, test, gpu, batchsize, model):

        self.test = test
        self.gpu = gpu
        self.batchsize = batchsize
        self.model = model

    def Confusion_matrix(self, save_path):

        val_results = {'y_pred': [], 'y_true': []}

        test_iter = chainer.iterators.SerialIterator(
            self.test, self.batchsize, repeat=False, shuffle=False)

        while True:
            X_test_batch = test_iter.next()
            X_test, y_test = concat_examples(X_test_batch, self.gpu)

            with chainer.no_backprop_mode(), chainer.using_config("train", False):
                y_pred = self.model.predictor(X_test)

            y_pred = chainer.cuda.to_cpu(y_pred.data)
            val_results['y_pred'].extend(np.argmax(y_pred, axis=1).tolist())
            val_results['y_true'].extend(y_test.tolist())

            if test_iter.is_new_epoch:
                test_iter.epoch = 0
                test_iter.current_position = 0
                test_iter.is_new_epoch = False
                test_iter._pushed_position = None
                break

        matrix = confusion_matrix(val_results['y_true'], val_results['y_pred'])
        save_matrix = save_path + "/"
        np.save(save_matrix + "matrix.npy", matrix)
        print(matrix)
