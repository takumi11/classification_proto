#! /usr/bin/env python
# -*- conding:utf-8 -*-
import matplotlib

import os
import datetime

from pathlib import Path

from util.args import parser
from util.get_dataset import get_dataset
from util.confusion_matrix import Confusion_Matrix
# from util.visualize import grad_cam
from models.four_lays_cnn import FOURLAYSCNN

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainercv.datasets import DirectoryParsingLabelDataset

matplotlib.use('Agg')


def main():

    args = parser()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = "../result/{}/".format(args.dataset) + now

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# datasets: {}'.format(args.dataset))
    print('')

    root = str(Path().resolve().parents[4] /
               'datasets' / 'processed' / args.dataset)
    datasets = DirectoryParsingLabelDataset(root)

    train, test, class_num, mean = get_dataset(datasets)
    print("finish load datasets !")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("made save directory !")

    model = L.Classifier(FOURLAYSCNN(class_num))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=save_dir)

    # print_trigger = (10, 'iteration')
    print_trigger = (1, 'epoch')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),
                   trigger=print_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=print_trigger))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                             'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                             'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss',
                                           'validation/main/loss', 'main/accuracy',
                                           'validation/main/accuracy', 'elapsed_time'
                                           ]), trigger=print_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    save_model = save_dir + "/" + "{}.model".format(now)
    chainer.serializers.save_npz(save_model, model)

    save_cf = save_dir + "/"
    cf = Confusion_Matrix(test, args.gpu, args.batchsize, model)
    cf.Confusion_matrix(save_cf)

    # grad_cam(model, test, args.gpu, save_dir)


if __name__ == '__main__':
    main()
