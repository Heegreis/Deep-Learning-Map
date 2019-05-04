# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_mnist_image(path, filename, type='train'):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0
    magic, num, rows, cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype='uint8')
        im = im.reshape(28, 28)
        im = Image.fromarray(im)
        if (type == 'train'):
            isExists = os.path.exists('dataset/MINIST_origin/train')
            if not isExists:
                os.mkdir('dataset/MINIST_origin/train')
            im.save('dataset/MINIST_origin/train/train_%s.bmp' % image, 'bmp')
        if (type == 'test'):
            isExists = os.path.exists('dataset/MINIST_origin/test')
            if not isExists:
                os.mkdir('dataset/MINIST_origin/test')
            im.save('dataset/MINIST_origin/test/test_%s.bmp' % image, 'bmp')


def load_mnist_label(path, filename, type='train'):
    full_name = os.path.join(path, filename)
    fp = open(full_name, 'rb')
    buf = fp.read()
    index = 0
    magic, num = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    Labels = np.zeros(num)

    for i in range(num):
        Labels[i] = np.array(struct.unpack_from('>B', buf, index))
        index += struct.calcsize('>B')

    if (type == 'train'):
        np.savetxt(
            'dataset/MINIST_origin/train_labels.csv',
            Labels,
            fmt='%i',
            delimiter=',')
    if (type == 'test'):
        np.savetxt(
            'dataset/MINIST_origin/test_labels.csv',
            Labels,
            fmt='%i',
            delimiter=',')

    return Labels


if __name__ == '__main__':
    path = 'dataset/MINIST_origin/'
    train_images = 'train-images.idx3-ubyte'
    load_mnist_image(path, train_images, 'train')
    train_labels = 'train-labels.idx1-ubyte'
    load_mnist_label(path, train_labels, 'train')
    test_images = 't10k-images.idx3-ubyte'
    load_mnist_image(path, test_images, 'test')
    test_labels = 't10k-labels.idx1-ubyte'
    load_mnist_label(path, test_labels, 'test')