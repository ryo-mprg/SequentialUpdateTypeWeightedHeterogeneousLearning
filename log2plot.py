# -*- coding: UTF-8 -*-
"""
log2plot
================================

ログファイルなどのテキストファイルからグラフを描きファイル保存する。

一定のパターンにマッチした行を抜き出し、その行が空白区切りで記述されている
と仮定して、指定された位置の数値を指定された順にプロットし、グラフをつくる。

使い方については、以下のコマンドで引数を確認すること。
::

    $ python log2plot.py -h
"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import argparse
import logging
from os import path
from matplotlib import pyplot


def parse_arguments():
    """
    :param --savefile:
        必須。
        保存するグラフのファイル名。フルパスまたは相対パスで指定すること。

    :param --logs:
        必須。
        ログファイル。フルパスまたは相対パスで指定すること。
        空白区切りで複数個指定できる。
        複数回指定した場合は、別の系列としてグラフにプロットされる。
        --logs で指定したログファイルの個数と
        --legends で指定した凡例の個数を一致させること。
        一致していないと、 AssertionError が発生する。

    :param --legends:
        必須。
        各系列の凡例のための文字列。空白区切りで複数個指定できる。
        空白区切りで指定するため、各凡例に空白を入れてはいけない。

    :param --keywords:
        必須。
        ログファイル中でプロット対象の行を抽出するためのキーワード。
        空白区切りで複数個指定できる。

    :param --x_column:
        オプション。
        プロット対象の行を空白区切りで分割したあと、
        xの値として使用するカラム番号をここで指定する。
        デフォルトは 1、つまり2番目。

    :param --y_column:
        オプション。
        プロット対象の行を空白区切りで分割したあと、
        yの値として使用するカラム番号をここで指定する。
        デフォルトは -1、つまり最後。

    :param --logscale:
        オプション。
        y軸を対数表示にする。

    :param --start:
        オプション。
        抽出したデータを何番目からプロットするか、指定する。
        学習では、初期のエラーがその後のエラーに比べてはるかに大きい
        ので、学習終盤の細かい変化を観察したいときは、このオプションを
        使って初期のエラーを除いておくと、みやすくなる可能性がある。

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--savefile',
        required=True,
        help=u'保存するグラフのファイル名'
    )
    parser.add_argument(
        '--logs',
        required=True,
        nargs='+',
        help=u"ログの相対パスまたは絶対パス"
    )
    parser.add_argument(
        '--legends',
        required=True,
        nargs='+',
        help=u'凡例用文字列'
    )
    parser.add_argument(
        '--keywords',
        required=True,
        nargs='+',
        help=u"プロット対象の行を抽出するためのキーワード"
    )
    parser.add_argument(
        '--x_column',
        type=int,
        default=1,
        help=u"xの値として使用するカラム(共通)"
    )
    parser.add_argument(
        '--y_column',
        type=int,
        default=-1,
        help=u"yの値として使用するカラム(共通)"
    )
    parser.add_argument(
        '--logscale',
        action='store_true',
        default=True,
        help=u'片対数(y)グラフにする'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help=u'プロット開始インデックス'
    )
    return parser.parse_args()


def setLogging():
    dateformat = '%m/%d %H:%M:%S'
    logformat = '%(asctime)s %(module)s:%(funcName)s:%(lineno)d %(message)s'
    logging.basicConfig(level=logging.INFO, format=logformat, datefmt=dateformat)


def log_args(args):
    args_dict = vars(args)
    for key in args_dict:
        logging.info('args:  {0}: {1}'.format(key, args_dict[key]))


def extract_xy(log_path, keywords, x_column, y_column, start, task_error, **kwargs):
    assert path.exists(log_path)
    with open(log_path) as f:
        lines = f.readlines()
        f.close()
    xs = []
    ys = []
    for l in lines:
        if task_error in l:
            words = l.split()
            xs.append(float(words[x_column]))
            ys.append(float(words[y_column]))
    x_base = 0
    for i in xrange(1, len(xs)):
        if 0 == xs[i]:
            x_base = xs[i-1]
        xs[i] += x_base
    return xs[start:], ys[start:]

"""
def save_text(task_error, xs, ys):
    xytext = '\n'.join(['{0}    {1}'.format(x, y) for x, y in zip(xs, ys)])
    save_text_file = task_error + '.txt'
    with open(save_text_file, 'w') as f:
        f.write(xytext)
        f.close()
        logging.info(save_text_file)
"""
def save_text2(task_error, xs):
    xytext = '\n'.join(['{0}'.format(x) for x in xs])
    save_text_file = log_path + 'text/' + task_error + '.txt'
    with open(save_text_file, 'w') as f:
        f.write(xytext)
        f.close()
        logging.info(save_text_file)

def main():
    args = parse_arguments()
    setLogging()
    log_args(args)

    assert len(args.legends) == len(args.logs)
    kwargs = vars(args)
    task_error = [
        'task_error_facial', 
        'task_error_gender', 
        'task_error_age', 
        'task_error_race', 
        'task_error_smile'
    ]

    for log in args.logs:
        #xs, ys = extract_xy(log, **kwargs)
        #save_text(log, xs, ys)
        for c in task_error:
            xs, ys = extract_xy(log, task_error = c, **kwargs)
            #save_text(c, xs, ys)
            save_text2(c, xs)


if __name__ == '__main__':
    main()
