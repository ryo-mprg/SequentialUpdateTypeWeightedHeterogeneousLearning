#-------------------------------------------------------------------------------
# Name:        piece.py
# Purpose:     必要だけど種々雑多なクラスや関数
#
# Author:      watasue
#
# Created:     24/03/2014
# Copyright:   (c) watasue 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import logging
from os import path

import numpy
import theano
from theano import tensor

import io
from log_tool import to_data_log

floatX = theano.config.floatX


class Parameter(object):
    """
    各レイヤーに所属する、学習すべきパラメータ。
    mini-batch gradient descent でパラメータを学習する。
    """
    def __init__(self, shape, name):
        """
        ゼロクリアされたパラメータを作成する。
        """
        value = numpy.zeros(shape, dtype=floatX)
        self.symbol = theano.shared(value=value, name=name, borrow=True)
        self.shape = shape

    def __str__(self):
        data_log = to_data_log(self.value)
        return '{0} {1}'.format(self.symbol, data_log)

    @property
    def value(self):
        'numpy.ndarray の実体を self.symbol から得る。'
        return self.symbol.get_value(borrow=True)

    @value.setter
    def value(self, value):
        assert value.shape == self.value.shape
        self.symbol.set_value(value, borrow=True)

    @property
    def filename(self):
        'パラメータの保存先。'
        return self.__filename

    def build_filename(self, prefix):
        """
        フォルダ名抜きの保存ファイル名をつくる
        """
        str_shape = 'x'.join('{}'.format(i) for i in self.value.shape)

        self.__filename = '{}_{}_{}.pkl'.format(
            prefix,
            str_shape,
            self.symbol.name
        )

    def init_value(self, rng, bound, bias=0):
        """
        乱数生成器によりパラメータを初期化する。
        """
        value = rng.uniform(low=-bound, high=bound, size=self.value.shape)
        value = value.astype(floatX) + bias
        self.symbol.set_value(value, borrow=True)

    def load(self, result_folder):
        """
        ファイルを読み、パラメータの値としてセットする。
        """
        if not hasattr(self, 'filename'):
            logging.warn('no filename')
            return
        filepath = path.join(result_folder, self.filename)
        if not path.exists(filepath):
            logging.warn('not exist: {0}'.format(filepath))
            return
        value = io.load(filepath)
        if value is None:
            logging.warn('value is None')
        elif value.shape != self.value.shape:
            logging.warn('{0} != {1}'.format(value.shape, self.value.shape))
        else:
            self.symbol.set_value(value, borrow=True)

    def save(self, result_folder):
        """
        パラメータをファイルに書く。
        """
        if hasattr(self, 'filename'):
            filepath = path.join(result_folder, self.filename)
            io.save(filepath, self.value)

    def push(self):
        """
        現在のパラメータを一時的に保存する。
        保存したパラメータに戻したいときには、 pop() を使う。
        """
        assert not hasattr(self, 'pushed')
        self.pushed = self.symbol.get_value()

    def pop(self):
        """
        push() で保存したパラメータに戻す。
        """
        assert hasattr(self, 'pushed')
        self.symbol.set_value(self.pushed)
        del self.pushed
