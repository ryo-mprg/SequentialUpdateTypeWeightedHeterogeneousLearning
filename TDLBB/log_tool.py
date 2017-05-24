#-------------------------------------------------------------------------------
# Name:        _logging.py
# Purpose:     ロギングのヘルパー関数
#
# Author:      watasue
#
# Created:     24/03/2014
# Copyright:   (c) watasue 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# -*- coding: UTF-8 -*-
import logging
import os
from os import path

import numpy


LOG_INDENT = '    '
# number of characters per line
N_CHAR_PER_LINE = 70

VISIBLE_CHAR_LEN = 35


def start_logging(logfile=None):
    """
    ロギングの設定
    この関数は、プログラムで、どの logging.*() が
    呼ばれるよりも早く呼ばれなければならない。

    logfile: ログの保存先。すでに存在していたら、追記。
    デフォルト(None)の場合、標準出力にだけログを表示。
    ファイル名が指定されていたら、標準出力にもログを表示。
    表示するのは、 Info 以上。
    """
    DATE_FORMAT = '%m/%d %H:%M:%S'
    LOG_FORMAT = '%(asctime)s %(module)s.py:%(lineno)d:%(funcName)s()\n'
    LOG_FORMAT += LOG_INDENT + '%(message)s'
    if logfile is None:
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT
        )
    else:
        result_folder = path.dirname(logfile)
        if not path.exists(path.dirname(logfile)):
            os.mkdir(result_folder)
        logging.basicConfig(
            filename=logfile,
            level=logging.INFO,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT
        )
        h = logging.StreamHandler(logging.sys.stdout)
        h.formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        logging.getLogger().addHandler(h)


def to_logging_message(messages):
    """
    messages: [str, ...]
    return: logging.info()に渡すと綺麗にログ出力してくれる
    """
    print_messages = []
    for m in messages:
        if len(m) < N_CHAR_PER_LINE:
            print_messages.append(m)
        else:
            head = m[:VISIBLE_CHAR_LEN]
            foot = m[-VISIBLE_CHAR_LEN:]
            print_messages.append(head + ' ... ' + foot)
    return ('\n' + LOG_INDENT).join(print_messages)


def to_data_log(data):
    """
    パラメータ・学習用サンプルなどのデータのロギング用文字列
    """
    min_value = float(numpy.min(data))
    max_value = float(numpy.max(data))
    fmt = '{0.dtype}{0.shape}  min: {1:.7e}  max: {2:.7e}'
    return fmt.format(data, min_value, max_value)
