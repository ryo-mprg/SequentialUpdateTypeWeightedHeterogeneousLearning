#-------------------------------------------------------------------------------
# Name:        _io.py
# Purpose:     ファイル入出力。データとパラメータが対象。
#
# Author:      watasue
#
# Created:     24/03/2014
# Copyright:   (c) watasue 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# -*- coding: UTF-8 -*-
import cPickle
import logging
from os import path

import numpy

from log_tool import to_logging_message
from log_tool import to_data_log

floatX = 'float32'

_ERRORS = (
    cPickle.PickleError,
    AttributeError,
    EOFError,
    ImportError,
    IndexError,
    IOError,
)

def save(filename, data):
    """
    非同期にファイルが更新される場合でも(比較的)安全にファイル書き込みを行う。
    書き込むデータは numpy.ndarray。
    """
    assert path.exists(path.dirname(filename)), filename
    while True:
        try:
            with open(filename, 'wb') as f:
                cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
                f.close()
                data_log = to_data_log(data)
                logging.info(to_logging_message([filename, data_log]))
                return
        except _ERRORS as e:
            logging.error(e)
            # セーブできるまで何度もトライする。


def load(filename):
    """
    非同期にファイルが更新される場合でも(比較的)安全にファイルを読み込む。
    読み込むデータは numpy.ndarray。
    ロードしたデータの加工とかは一切しない。
    """
    assert path.exists(filename), filename
    while True:
        try:
            with open(filename, 'rb') as f:
                data = cPickle.load(f)
                f.close()
                data_log = to_data_log(data)
                logging.info(to_logging_message([filename, data_log]))
                return data
        except _ERRORS as e:
            logging.error(e)
            # ロードできるまで何度もトライする。
