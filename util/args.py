#! /usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from models import archs


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--batchsize', '-b', type=int, default=100)
    parser.add_argument('--dataset', '-d', default='normal')
    parser.add_argument('--frequency', '-f', type=int, default=-1)
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='4-lay')

    args = parser.parse_args()

    return args
