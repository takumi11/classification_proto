#! /usr/bin/env python
# -*- conding:utf-8 -*-
import matplotlib

import os
import datetime

from pathlib import Path

from util.args import parser
from util.get_dataset import get_dataset
from util.confusion_matrix import Confusion_Matrix
from util.print_txt import PrintTXT
# from util.visualize import grad_cam
from models import archs

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

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("made save_directory !")

    log = PrintTXT()
    log.add('# GPU: {}'.format(args.gpu))
    log.add('# Minibatch-size: {}'.format(args.batchsize))
    log.add('# epoch: {}'.format(args.epoch))
    log.add('# datasets: {}'.format(args.dataset))
    log.add('# using model: {}'.format(args.arch))
    log.add('')
    log.save(save_dir + "/log.txt")

    root = str(Path().resolve().parents[4] /
               'datasets' / 'processed_316' / args.dataset)
    datasets = DirectoryParsingLabelDataset(root)

    train, test, class_num, mean = get_dataset(datasets)
    print("finish load datasets !")

    model = L.Classifier(archs[args.arch](class_num))
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0003))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=save_dir)

    # print_trigger = (10, 'iteration')
    log_trigger = (1, 'epoch')

    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=chainer.training.triggers.ManualScheduleTrigger(
                       [100, 200, 250, 300, 350, 400], 'epoch'))
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu),
                   trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=log_trigger))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch', file_name='loss.png', marker='.'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png', marker='.'))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr'
        ]), trigger=log_trigger)
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
