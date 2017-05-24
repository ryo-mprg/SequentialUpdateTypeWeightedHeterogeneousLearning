# -*- coding: UTF-8 -*-
"""
visualize_face_feature
================================

"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import argparse
import imp
import logging
import os
from os import path

import cv2
import numpy
import math

from TDLBB import layer
from TDLBB import io
from TDLBB.log_tool import start_logging
from TDLBB.log_tool import to_logging_message

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
    logging.info(info)

    return imp.load_module(module_name, *info)

def plot(canvas, rows, cols, data, color):
    xs = data[:, 0] * cols
    ys = data[:, 1] * rows
    positions = zip(xs, ys)
    for pos in positions:
        cv2.circle(canvas, pos, 3, color, -1 )
        cv2.circle(canvas, pos, 3, (0, 0, 0,), 1 )


def plot2(canvas, rows, cols, data, actual_a, actual_r, actual_s, color):
   
    xs = [0]*5
    ys = [0]*5
    for i in xrange(5):
       xs[i] = int(data[i][0]*1000000) * cols / 1000000
       ys[i] = int(data[i][1]*1000000) * rows / 1000000
    positions = zip(xs, ys)
    

    #器官点描写
    for pos in positions:
        cv2.circle(canvas, pos, 3, color, -1 )  
        cv2.circle(canvas, pos, 3, (0, 0, 0,), 1 )

    #年齢描写
    font = cv2.FONT_HERSHEY_PLAIN
    actual_a = int(actual_a * 10) / 10
    cv2.putText(canvas, str(actual_a), (10, 10), font, 1.5, (0, 255, 0))

    #人種描写
    if actual_r == 0 : race = 'Asian'
    if actual_r == 1 : race = 'White'
    if actual_r == 2 : race = 'Black'
    cv2.putText(canvas, race, (10, 90), font, 1.5, (0, 255, 0))

    #笑顔度描写
    cv2.putText(canvas, str(actual_s), (10, 50), font, 1.5, (0, 255, 9))



def visualize(in_data, actual, expect, gender_max, actual_a, actual_r, actual_s, savefile):
    logging.info('in_data: {}'.format(in_data.shape))

    rows, cols = in_data.shape[-2:]

    if 3 == in_data.ndim:
        canvas = in_data
    else:
        canvas = cv2.cvtColor(in_data, cv2.cv.CV_GRAY2BGR)

#教師
    #plot(canvas, rows, cols, expect / 255., (0, 0, 255,))

#検出(B, G, R)
    if gender_max == 0:
       plot2(canvas, rows, cols, actual, actual_a, actual_r, actual_s, (0, 0, 255,))
    else :
       plot2(canvas, rows, cols, actual, actual_a, actual_r, actual_s, (255, 0, 0,))

    cv2.imwrite(savefile, canvas)
    logging.info('saved: {0}'.format(savefile))


def main():
    args = parse_arguments()
    result_folder = path.dirname(args.config_path)
    start_logging(path.splitext(args.config_path)[0] + '.log')
    cfg_mod = load_module(args.config_path)


    data_set = cfg_mod.DATA_GENERATOR.ndarray_dict
    network = cfg_mod.NETWORK
    network.load_param(result_folder)
    function = network.get_predictor()

    # 評価時の年齢の間違いの許容誤差
    # 5なら±5歳まで間違えてもOKという意味
    th = 5

    # 認識できたサンプルのカウント変数
    TrueCount_a = 0
    FalseCount_a = 0
    TrueCount_g = 0
    FalseCount_g = 0
    TrueCount_r = 0
    FalseCount_r = 0
    TrueCount_s = 0
    FalseCount_s = 0
    err_t = [0, 0, 0, 0, 0, 0]
    total = 0
    actual_k = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    # ネットワークの出力と教師信号との比較
    for i, input_data in enumerate(data_set[cfg_mod.INPUT_KEY]):
        savefile = path.join(result_folder, 'png', 'visualize_{0:05d}.png'.format(i))
        actual = function([input_data])[0]
        expect_k = data_set[cfg_mod.feature_positions][i]
        expect_g = data_set[cfg_mod.gender][i]
        expect_a = data_set[cfg_mod.age][i]
        expect_r = data_set[cfg_mod.race][i]
        expect_s = data_set[cfg_mod.smile][i]

        total = total + 1        
        print total

   
        #器官点出力用の設定
        for a in range(5) :
           for b in range(2) :
              actual_k[a][b] = 1 / (1 + math.exp(-actual[a * 2 + b]))

        #性別
        actual_g = [0, 0]
        gender_max = 0
        total_exp = 0.0
        for j in range(10,12) :
           total_exp = total_exp + math.exp(actual[j])
        for j in range(0,2) :
           actual_g[j] = math.exp(actual[j + 10]) / total_exp
        for j in range(0,2) :
           if actual_g[gender_max] < actual_g[j] :
              gender_max = j

        #年齢
        actual_a = 1 / (1 + math.exp(-actual[12]))
        actual_a = actual_a * 66.0

        #人種
        actual_r = [0, 0, 0]
        race_max = 0
        total_exp = 0.0

        for j in range(13,16) :
           total_exp = total_exp + math.exp(actual[j])
        for j in range(0,3) :
           actual_r[j] = math.exp(actual[j + 13]) / total_exp
        for j in range(0,3) :
           if actual_r[race_max] < actual_r[j] :
              race_max = j

        #笑顔度
        actual_s =  1 / (1 + math.exp(-actual[16]))
        actual_s = actual_s * 100.0

        #識別結果の描画
        #if total == 500 :break
            #visualize(input_data, actual_k, expect_k, gender_max, actual_a, race_max,actual_s, savefile)


        #評価

        #器官点
        for c in range(0, 10):
	    actual[c] = 1 / (1 + math.exp(-actual[c]))
        interocluer = numpy.linalg.norm(expect_k[0] - expect_k[1])
        err = numpy.zeros(5)
        for c in range(5):
            err[c]=(math.sqrt((expect_k[c][0]-actual[c*2]*255.)**2+(expect_k[c][1]-actual[c*2+1]*255.)**2))/interocluer#/1.354
	    if err[c] <= 0.1 : err_t[c] = err_t[c] + 1

        #性別
        if expect_g[0] == 1 : gen = 0
        else : gen = 1
        if gender_max == gen : TrueCount_g += 1
        else : FalseCount_g += 1

        #性別
        if math.fabs(expect_a - actual_a) <= 5.0 : TrueCount_a += 1
        else : FalseCount_a += 1

        #人種
        if expect_r[0] == 1 : race = 0
        if expect_r[1] == 1 : race = 1
        if expect_r[2] == 1 : race = 2

        if race == race_max : TrueCount_r += 1
        else : FalseCount_r += 1

        #笑顔度
        if math.fabs(expect_s - actual_s) <= 10.0 : TrueCount_s += 1
        else : FalseCount_s += 1


    #識別精度の算出と表示
    rate_left_eye = float(err_t[0]) / total
    rate_right_eye = float(err_t[1]) / total
    rate_nose = float(err_t[2]) / total
    rate_left_mouth = float(err_t[3]) / total
    rate_right_mouth = float(err_t[4]) / total

    rate_positions = [
        rate_left_eye, 
        rate_right_eye, 
        rate_nose, 
        rate_left_mouth, 
        rate_right_mouth
    ]
    
    rate_facial =  sum(rate_positions) / len(rate_positions)

    rate_gender =  float(TrueCount_g) / (TrueCount_g + FalseCount_g)
    rate_age =  float(TrueCount_a) / (TrueCount_a + FalseCount_a)
    rate_race =  float(TrueCount_r) / (TrueCount_r + FalseCount_r)
    rate_smile = float(TrueCount_s) / (TrueCount_s + FalseCount_s)

    rate_hetero = [
        rate_facial, 
        rate_gender, 
        rate_age, 
        rate_race, 
        rate_smile
    ]

    all_ave = sum(rate_hetero) / len(rate_hetero)

    
    #print 'time :'time.clock()
    print '-- feature positions Regression Rate ----------'
    print 'left eye    :', rate_left_eye
    print 'right eye   :', rate_right_eye
    print 'nose        :', rate_nose
    print 'left mouth  :', rate_left_mouth
    print 'right mouth :', rate_right_mouth
    print 'average     :', rate_facial
    print '-----------------------------------------------'


    print 'gender Recognition Rate :', rate_gender
    print 'age Regression Rate     :', rate_age
    print 'race Recognition Rate   :', rate_race
    print 'smile Regression Rate   :', rate_smile
    
    
    print 'All average             :', all_ave


if __name__ == '__main__':
    main()

