# -*- coding: UTF-8 -*-
"""
pkl_shape
================================

*.pkl を読み、 numpy.ndarray の shape を出力する

使い方がわからない場合は、以下のコマンドで引数を確認すること。
::

    $ python pkl_shape.py -h
"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import argparse
import cPickle
import logging

import numpy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path', help='a pkl file path(*.pkl)')
    return parser.parse_args()


def set_logging():
    dateformat = '%m/%d %H:%M:%S'
    logformat = '%(asctime)s %(module)s:%(funcName)s:%(lineno)d\n'
    logformat += '    %(message)s'
    logging.basicConfig(level=logging.INFO, format=logformat, datefmt=dateformat)


def log_args(args):
    """
    parse_arguments() の解析結果
    """
    args_dict = vars(args)
    for key in args_dict:
        logging.info('{0}: {1}'.format(key, args_dict[key]))


def get_log_message(afile, data):
    """
    セーブ・ロード完了時のロギング用文字列
    """
    min_value = numpy.min(data)
    max_value = numpy.max(data)
    formatter = '{0}  shape:{1}  min:{2} max:{3}'
    return formatter.format(afile, data.shape, min_value, max_value)


def load(loadfile):
    """
    データをロードする。
    ロードしたデータの加工とかは一切しない。
    """
    floatX = 'float32'
    data = None
    while data is None:
        try:
            with open(loadfile, 'rb') as f:
                data = cPickle.load(f).astype(floatX)
                f.close()
                logging.info(get_log_message(loadfile, data))
        except EOFError:
            logging.info( 'EOFError: {0}'.format(loadfile))
    return data


def main():
    args = parse_arguments()
    set_logging()
    log_args(args)

    load(args.pkl_path)


if __name__ == '__main__':
    main()