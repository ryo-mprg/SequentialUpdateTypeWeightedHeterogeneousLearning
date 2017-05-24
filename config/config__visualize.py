# -*- coding: UTF-8 -*-
"""
config__fine_tuning
================================

学習に関する全ての設定
"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

from glob import glob
from os import path

import numpy

from TDLBB import layer
from TDLBB.model import DataGenerator
from TDLBB.model import Network
from TDLBB.model import UpdateCounter
from TDLBB.node import DataNode
from TDLBB.node import Node

# 予測なのでバッチサイズは 1
BATCH_SIZE = 1
INPUT_CHANNEL = 1
INPUT_ROWS = 100
INPUT_COLS = 100

OUTPUT_NODES = 17

INPUT_KEY = 'x'
feature_positions = 'f'
gender = 'g'
age = 'a'
race = 'r'
smile = 's'

# 学習データに対応するノード
TRAIN_X_FILES = glob(path.join('..', 'pkl', 'test', '*_x.pkl'))
TRAIN_F_FILES = glob(path.join('..', 'pkl', 'test', '*_feature_positions.pkl'))
TRAIN_G_FILES = glob(path.join('..', 'pkl', 'test', '*_gender.pkl'))
TRAIN_A_FILES = glob(path.join('..', 'pkl', 'test', '*_age.pkl'))
TRAIN_R_FILES = glob(path.join('..', 'pkl', 'test', '*_race.pkl'))
TRAIN_S_FILES = glob(path.join('..', 'pkl', 'test', '*_smile.pkl'))


TRAIN_NODE_X = DataNode(
  (BATCH_SIZE, INPUT_ROWS, INPUT_COLS),
  INPUT_KEY,
  TRAIN_X_FILES,
)
TRAIN_NODE_F = DataNode(
  (BATCH_SIZE, 5, 2),
  feature_positions,
  TRAIN_F_FILES,
)
TRAIN_NODE_G = DataNode(
  (BATCH_SIZE,2,),
  gender,
  TRAIN_G_FILES,
)
TRAIN_NODE_A = DataNode(
  (BATCH_SIZE,),
  age,
  TRAIN_A_FILES,
)
TRAIN_NODE_R = DataNode(
  (BATCH_SIZE,3,),
  race,
  TRAIN_R_FILES,
)
TRAIN_NODE_S = DataNode(
  (BATCH_SIZE,),
  smile,
  TRAIN_S_FILES,
)


DATA_GENERATOR = DataGenerator(
  TRAIN_NODE_X, 
  TRAIN_NODE_F, 
  TRAIN_NODE_G, 
  TRAIN_NODE_A, 
  TRAIN_NODE_R, 
  TRAIN_NODE_S
)


# ネットワーク構造(アーキテクチャ)
_conv1 = [
  layer.Reshape(1, 100, 100),
  layer.Normalize(),
  layer.Convolution((16, 9, 9), layer.Plain(), tune=True),
  layer.MaxOut(2),
  layer.MaxPool(2, 2),
]

_conv2 = [
  layer.Normalize(),
  layer.Convolution((32, 9, 9), layer.Plain(), tune=True),
  layer.MaxOut(2),
  layer.MaxPool(2, 2),
]


_conv3 = [
  layer.Normalize(),
  layer.Convolution((64, 9, 9), layer.Plain(), tune=True),
  layer.MaxOut(2),
  layer.Flatten(),
]

_full1 = [
  layer.Normalize(),
  layer.Linear(0.5, 0),
  layer.Full(200, layer.Sigmoid(), tune=True),
]

'''
_coords_layer = [
   layer.Reshape(6, 2),
   layer.Regression(TRAIN_NODE_YC)
]
_sex_layer = [
   layer.Sigmoid(),
   layer.Regression(TRAIN_NODE_YS)
]
_age_layer = [
   layer.SoftMax(),
   layer.Regression(TRAIN_NODE_YA)
]
'''
_out = [
    layer.Normalize(),
    layer.Full(OUTPUT_NODES, layer.Plain(), tune=True),
    layer.GetRange(0, 17),
]

_layers = [
    _conv1,
    _conv2,
    _conv3,
    _full1,
    _out
]

NETWORK = Network(
    DATA_GENERATOR.nodes[INPUT_KEY],
    _layers,
    DATA_GENERATOR.nodes[INPUT_KEY],
)

MAX_SAVE_FILE = 10
