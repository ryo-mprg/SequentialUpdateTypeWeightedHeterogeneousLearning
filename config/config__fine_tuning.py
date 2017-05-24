# -*- coding: UTF-8 -*-
"""
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
from TDLBB.update import BaseUpdateCounter
from TDLBB.update import UpdateCounter
from TDLBB.update import AnnealingUpdateCounter
from TDLBB.update import ThoroughlyUpdateCounter
from TDLBB.update import SelectiveUpdateCounter
from TDLBB.node import DataNode
from TDLBB.node import Node

BATCH_SIZE = 10
epoch = 4400

INPUT_KEY = 'x'
feature_positions = 'f'
gender = 'g'
age = 'a'
race = 'r'
smile = 's'

INPUT_CHANNELS = 10
INPUT_ROWS = 100
INPUT_COLS = 100
INPUT_NODES = INPUT_CHANNELS * INPUT_ROWS * INPUT_COLS

OUTPUT_NODES = 17

ROOT_FOLDER = path.dirname(__file__)

TRAIN_X_FILES = glob(path.join('..', 'pkl', 'train', '*_x.pkl'))
TRAIN_F_FILES = glob(path.join('..', 'pkl', 'train', '*_feature_positions.pkl'))
TRAIN_G_FILES = glob(path.join('..', 'pkl', 'train', '*_gender.pkl'))
TRAIN_A_FILES = glob(path.join('..', 'pkl', 'train', '*_age.pkl'))
TRAIN_R_FILES = glob(path.join('..', 'pkl', 'train', '*_race.pkl'))
TRAIN_S_FILES = glob(path.join('..', 'pkl', 'train', '*_smile.pkl'))

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
  (BATCH_SIZE, 2,),
  gender,
  TRAIN_G_FILES,
)
TRAIN_NODE_A = DataNode(
  (BATCH_SIZE,),
  age,
  TRAIN_A_FILES,
)
TRAIN_NODE_R = DataNode(
  (BATCH_SIZE, 3,),
  race,
  TRAIN_R_FILES,
)

TRAIN_NODE_S = DataNode(
  (BATCH_SIZE,),
  smile,
  TRAIN_S_FILES,
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
  layer.DropOut(0.5),
  layer.Full(200, layer.Sigmoid(), tune=True),
]


# 器官点
_feature_positions_layers = [
  layer.Reshape(5, 2),
  layer.Sigmoid(),
  layer.Regression_facial(TRAIN_NODE_F/255.),
]

# 性別
_gender_layers = [
  layer.SoftMax(),
  layer.MultiClass_gender(TRAIN_NODE_G),
]

# 年齢
_age_layers = [
  layer.Sigmoid(),
  layer.Regression_age(TRAIN_NODE_A/66.),
]

#人種
_race_layers = [
  layer.SoftMax(),
  layer.MultiClass_race(TRAIN_NODE_R),
]

#笑顔度
_smile_layers = [
  layer.Sigmoid(),
  layer.Regression_smile(TRAIN_NODE_S/100.),
]

# 出力層
_out = [
  layer.Normalize(),
  layer.Full(OUTPUT_NODES, layer.Plain(), tune=True),
  #layer.MaxOut(2),
  layer.WeightedHeterogeneous(
      (10,_feature_positions_layers),
      (2, _gender_layers),
      (1, _age_layers),
      (3, _race_layers),
      (1, _smile_layers),
  ),
]

_layers = [
  _conv1, 
  _conv2, 
  _conv3, 
  _full1, 
  _out
]

DATA_GENERATOR = DataGenerator(
  TRAIN_NODE_X, 
  TRAIN_NODE_F, 
  TRAIN_NODE_G, 
  TRAIN_NODE_A, 
  TRAIN_NODE_R, 
  TRAIN_NODE_S
)

NETWORK = Network(
  DATA_GENERATOR.nodes[INPUT_KEY],
  _layers,
  DATA_GENERATOR.nodes.values()
)

# 学習の更新回数・学習率
COUNTER = UpdateCounter(
      max_update_count = 250 * epoch,
      learning_rate = 0.1,
      epoch = epoch
)
