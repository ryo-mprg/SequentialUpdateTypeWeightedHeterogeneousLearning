# -*- coding: UTF-8 -*-
"""
学習をする(何でも)
================================

マルチクラス識別だろうが、回帰だろうが、二値画像だろうが、
層別学習だろうが何でも学習できるプログラム。

"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import argparse
import imp
import logging
import os
import tarfile
import time
from glob import glob
from os import path

import numpy

from TDLBB.log_tool import start_logging



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        help='configuration file path'
    )

    return parser.parse_args()


def load_module(module_path):
    """
    module_path: モジュールへのパス(フォルダ・拡張子含む)

    return ロードされたモジュール
    """
    head, tail = path.split(module_path)
    module_name = path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    #logging.info(info)
    
    return imp.load_module(module_name, *info)


def make_tgz(counter, result_folder):
    """
    *.{log,pkl} をかためた *.tgz ファイルにする。
    tgz ファイルの basename は年月日時分秒が使われるので、
    かぶることはまずない。
    """
    head = time.strftime('%Y-%m-%d_%H-%M-%S')
    tail = '_{0:08d}.tgz'.format(counter.count)
    bname = head + tail
    tgzfile = path.join(result_folder, bname)
    with tarfile.open(tgzfile, 'w:gz') as tar:
        logs = glob(path.join(result_folder, '*.log'))
        for logfile in logs:
            tar.add(logfile)
        pkls = glob(path.join(result_folder, '*.pkl'))
        for pklfile in pkls:
            tar.add(pklfile)
        logging.info(bname)
        tar.close()


def train_loop(batch_size, counter, data_getter, net, result_folder):
    """
    学習のループ。
    パラメータ更新が既定回数に達したら、 return で抜ける。
    """
    fmt = '{0}  train_error_hetero: {1:.7e}'
    fmt_list = [
        '{0}  task_error_facial: {1:.7e}', 
        '{0}  task_error_gender: {1:.7e}', 
        '{0}  task_error_age: {1:.7e}', 
        '{0}  task_error_race: {1:.7e}', 
        '{0}  task_error_smile: {1:.7e}'
    ]   
    task_error = [[],[],[],[],[]]

    while(True):
        data_set = data_getter.next()
        learning_rate = counter.get_learning_rate(net, data_set)
        functions = net.get_trainer(learning_rate, data_set)
        
        for i in xrange(0, data_getter.n_data, batch_size):
            
            for function in functions:
                error = function(i) # momentum のように複数呼び出しに対応
            counter.update(error)

            if counter.is_over():
                """
                学習終了
                """
                net.save_param(result_folder)
                logging.info(fmt.format(counter, float(error[1])))
                for i in xrange(0, len(fmt_list)):
                    logging.info(fmt_list[i].format(counter, float(error[2][i])))
                    task_error[i].append(float(error[2][i]))
                return

            if counter.is_epoch():
                """
                重みの更新
                """
                net.save_param(result_folder)
                net.update_weight(task_error)

                rng = numpy.random.RandomState()
                net.preprocess(rng, result_folder)
                make_tgz(counter, result_folder)


            logging.info(fmt.format(counter, float(error[1])))
            for i in xrange(0, len(fmt_list)):
                logging.info(fmt_list[i].format(counter, float(error[2][i])))
                task_error[i].append(float(error[2][i]))


def main():
    args = parse_arguments()
    result_folder = path.dirname(args.config_path)
    start_logging(path.splitext(args.config_path)[0] + '.log')
    cfg_mod = load_module(args.config_path)

    rng = numpy.random.RandomState()
    cfg_mod.NETWORK.preprocess(rng, result_folder)
    make_tgz(cfg_mod.COUNTER, result_folder)
    
    train_loop(
        cfg_mod.BATCH_SIZE,
        cfg_mod.COUNTER,
        cfg_mod.DATA_GENERATOR,
        cfg_mod.NETWORK,
        result_folder,
    )
    
    make_tgz(cfg_mod.COUNTER, result_folder)
    logging.info('OVER')


if __name__ == '__main__':
    main()
