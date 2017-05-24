#-------------------------------------------------------------------------------
# Name:        node.py
# Purpose:
#
# Author:      watasue
#
# Created:     25/03/2014
# Copyright:   (c) watasue 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# -*- coding: UTF-8 -*-
import logging

import numpy
import theano
from theano import tensor

import io
from _piece import floatX


class Node(object):
    """
    形状(numpy.ndarray.shape)とシンボル(theano.variable)と
    dtype(numpy.array.dtype)をまとめたもの。

    可視層のノード(入力データ)にもなるし、
    階層をたどっていって出力するノードにもなる。
    また、出力の真値のノードにもなる。
    """
    def __init__(self, shape, symbol=None, name=''):
        """
        shape: バッチサイズを含めたデータの形状。
        fully-connected layer ならたとえば (10, 100) とかになる。
        この場合は、バッチサイズは10, ノード数は 100。
        convolutional layer ならたとえば、 (10, 50, 104, 32) とかになる。
        この場合は、バッチサイズは10, マップ数は50, 行数は104, 列数は32。

        symbol: theano.variable
        レイヤーの最初のノードは、None でもよい。その場合は shape から自動的に
        計算される。
        """
        self.shape = tuple(shape[:])
        self.symbol = symbol
        self.name = name
        self.size = numpy.prod(self.shape)
        self.ndim = len(self.shape)
        self.batch_size = self.shape[0]
        self.__shape_to_symbol()
        self.node = self # _2in1 で使えるようにするため

    def __str__(self):
        fmt = '{0}:{1}  {2}  '
        class_name = self.__class__.__name__
        result = fmt.format(class_name, self.name, self.size)
        if 1 < len(self.shape):
            result += ' x '.join(str(i) for i in self.shape)
        else:
            result += str(self.shape[0])
        return result

    def __mul__(self, coeff):
        """
        Node(node0 * 255.) ができるようにする
        """
        return Node(self.shape, self.symbol * coeff)

    def __div__(self, coeff):
        """
        Node(node0 / 255.) ができるようにする
        """
        return Node(self.shape, self.symbol / coeff)

    def __shape_to_symbol(self):
        """
        shape から判別して symbol を定義する。
        """
        if not self.symbol is None:
            return
        symbol_dict = {
            1: tensor.fvector,
            2: tensor.fmatrix,
            3: tensor.ftensor3,
            4: tensor.ftensor4,
        }
        self.symbol = symbol_dict[self.ndim](name=self.name)


class DataNode(Node):
    """
    Node の中で、ファイル保存されたデータと対応するノード。
    """
    def __init__(self, shape, name, files, partial=None):
        """
        :param shape: バッチサイズを含めたデータの形状。 Node と同様。
        :param name: 名前
        :param files: データファイルのリストまたはタプル
        :param partial: データを部分的に使用する場合にここに
        (start, end, step)を記述する。
        start は開始インデックス(含む)、endは終了インデックス(含まない)、
        step はインデックスのジャンプステップ。
        例えば、 partial=(0,10,2) とすると、使用するデータは、
        [0,2,4,6,8] になる。
        デフォルトは None。None のときは全データを使う。
        """
        self.partial = partial
        if partial is not None:
            shape = list(shape)
            shape[1] = (partial[1] - partial[0]) / partial[2]
        Node.__init__(self, shape, name=name)
        assert 0 < len(files), 'NO FILES'
        self.files = files
        self.files.sort()
        self.__init_shared()
        logging.info(self)

    def __str__(self):
        fmt = '{0} files:{1}'
        return fmt.format(Node.__str__(self), len(self.files))

    def __get_valid_value(self, value):
        if self.partial is not None:
            start, end, step = self.partial
            return value[:, start:end:step]
        else:
            return value

    def __init_shared(self):
        """
        theano.shared() をつくっておく。
        学習中は、 set_value(borrow=True) で値を更新する。
        毎回 theano.shared() をつくるとメモリが異常に消費されるから。
        """
        value = io.load(self.files[0]).astype(floatX)
        value = self.__get_valid_value(value)
        self.n_data = len(value)
        self.shared = theano.shared(value, self.name, borrow=True)

    def update(self, index):
        """
        self.files[index] のデータを読み、保持しているデータを更新する。
        """
        value = io.load(self.files[index]).astype(floatX)
        value = self.__get_valid_value(value)
        assert value.shape == self.shared.get_value(borrow=True).shape
        self.shared.set_value(value, borrow=True)
