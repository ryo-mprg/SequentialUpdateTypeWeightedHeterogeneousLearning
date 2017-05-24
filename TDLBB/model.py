#-------------------------------------------------------------------------------
# Name:        _model.py
# Purpose:     Node + Layer[] で構成されるネットワーク
#
# Author:      watasue
#
# Created:     24/03/2014
# Copyright:   (c) watasue 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# -*- coding: UTF-8 -*-
import logging
from collections import OrderedDict

import numpy
import theano
from theano import tensor

import layer
from log_tool import to_logging_message
from log_tool import to_data_log
from node import Node
from _piece import floatX
from update import UpdateCounter # 後方互換のため
from calc_weight import calc

def __is_valid_type(obj, types):
    """
    :param obj: 検査対象のオブジェクト
    :param types: obj は typesに列記した型のいずれかであるはず
    :return: 真ならば、obj は typesに列記した型のいずれか
    """
    return type(obj) in types


def get_layer_generator(layers, required_attr=None):
    """
    特定の属性を持つ layer を順に返すジェネレータ

    :param layers:
        [[layer, ..], ..] という構造(layer._Base のリストのリスト)

    :param required_attr:
        layer のうち、この属性を持つものだけを返す
        デフォルトでは、無条件に返す
    """
    assert __is_valid_type(layers, (list, tuple)), type(layers)
    for ilayer in layers:
        assert __is_valid_type(ilayer, (list, tuple)), type(ilayer)
        for sub_layer in ilayer:
            assert isinstance(sub_layer, layer._Base), sub_layer
            if required_attr is None:
                yield sub_layer
            elif hasattr(sub_layer, required_attr):
                yield sub_layer
            else:
                pass


class Network(object):
    """
    Node() と [layer, ...] の組。
    """
    def __init__(self, input_node, layers, terminal_nodes=[]):
        self.input_node = input_node
        if isinstance(terminal_nodes, Node):
            terminal_nodes = [terminal_nodes]
        self.terminal_symbols = [n.symbol for n in terminal_nodes]
        self.layers = layers
        self.__set_param_prefix()
        self.weight = calc()
        self.__chain()

    def __chain(self):
        """
        シンボリックモデルを連鎖させて、 front propagation を計算する。
        """
        node = self.input_node
        messages = []
        gen = get_layer_generator(self.layers)
        weights = [self.weight['facial'], self.weight['gender'], self.weight['age'], self.weight['race'], self.weight['smile']]
        for l in gen:
            messages += [str(node)]
            node = l.chain(node, weights)
            messages += [str(l), str(node)]
        logging.info(to_logging_message(messages))
        self.output_node = node
    
    def __set_param_prefix(self):
        """
        各 _Parametric 層の保存ファイル名を設定する
        """
        sn = 0 # serial number
        gen = get_layer_generator(self.layers, 'set_prefix')
        for l in gen:
            l.set_prefix(sn)
            sn += 1

    @property
    def param2tune(self):
        """
        更新すべきパラメータ(theano.variable)のリスト
        """
        result = []
        for l in get_layer_generator(self.layers, 'tune'):
            if l.tune:
                assert hasattr(l, 'params')
                result += [p.symbol for p in l.params.values()]
        return result

    def init_param(self, rng):
        """
        全レイヤーのパラメータを初期化する。
        """
        for l in get_layer_generator(self.layers, 'init_param'):
            l.init_param(rng)
        logging.info(self)

    def __str__(self):
        messages = []
        for l in get_layer_generator(self.layers, 'params'):
            messages += [str(p) for p in l.params.values()]
        if 0 < len(messages):
            return to_logging_message(messages)
        else:
            return repr(self)

    def load_param(self, result_folder):
        """
        保存されているファイルから、パラメータの値を読み込む。

        result_folder: 保存先のフォルダ名
        """
        for l in get_layer_generator(self.layers, 'load'):
            l.load(result_folder)

    def save_param(self, result_folder):
        """
        パラメータをファイル保存する。

        result_folder: 保存先のフォルダ名
        """
        for l in get_layer_generator(self.layers, 'save'):
            l.save(result_folder)

    def push_param(self):
        """
        現在のパラメータを一時的に保存する。
        保存したパラメータに戻したいときには、 pop_param() を使う。
        """
        for l in get_layer_generator(self.layers, 'push'):
            l.push()

    def pop_param(self):
        """
        push() で保存したパラメータに戻す。
        """
        for l in get_layer_generator(self.layers, 'pop'):
            l.pop()

    def get_predictor(self):
        """
        推定をする theano.function を得る。
        """
        return theano.function(
            inputs=self.terminal_symbols,
            outputs=self.output_node.symbol
        )

    def calc_cost(self):
        if 1 != self.output_node.ndim:
            return
        self.cost = self.output_node.symbol.sum() / self.input_node.batch_size
        self.cost_hetero = self.output_node.symbol.sum() / self.output_node.size
        self.cost_each = self.output_node.symbol.sum(axis=0) / self.input_node.batch_size# / self.output_node.shape[-1]

    def calc_grads(self, wrt):
        """
        wrt: stands for "With Respect To".
        何に対する偏微分かを指定する。
        """
        grads = theano.grad(self.cost, wrt=wrt)
        self.grad_map = zip(wrt, grads)
        assert 0 < len(self.grad_map), (wrt, grads)

    def get_trainer(self, learning_rate, data_dict):
        """
        :param learning_rate: 学習率。焼き鈍しによる変化があるかも。

        :param data_dict: str をキーとし、
        学習用データセットを値とする。値は theano.shared()。

        :return: バッチサイズごとのエラーの平均。
        """
        updates = OrderedDict()
        for param_i, grad_i in self.grad_map:
            # 勾配が nan になる場合がある。そのときはパラメータを更新しない。
            updates[param_i] = tensor.switch(
                tensor.isnan(grad_i),
                param_i,
                param_i - learning_rate * grad_i
            )
        index = tensor.iscalar()
        givens = OrderedDict()
        for k, v in data_dict.items():
            assert k in self.terminal_symbols
            givens[k] = v[index : index + self.input_node.batch_size]
        # NetworkWithMomentum が複数の theano.function を返すので、
        # train.py のコードを簡素に保つため、こっちもタプルを返す

        return (theano.function(
            inputs=[index], outputs=[self.cost, self.cost_hetero, self.cost_each], updates=updates, givens=givens
        ),)

    def preprocess(self, rng, result_folder):
        """
        """
        self.init_param(rng)
        self.load_param(result_folder)
        self.save_param(result_folder)
        self.calc_cost()
        self.calc_grads(self.param2tune)

    def update_weight(self, task_error):
        self.weight.calc_weight(task_error)
        self.__chain()

class NetworkWithMomentum(Network):
    """
    momentum で学習できる Network

    Classical Momentum(CM) とも言われている手法。

    Polyak, B.T. Some methods of speeding up the convergence
    of iteration methods. USSR Computational Mathematics
    and Mathematical Physics, 4(5):1{17, 1964.
    """
    def __init__(
        self, input_node, layers, terminal_nodes=[], momentum_coeff=0.999
    ):
        """
        :param momentum_coeff: モーメントの係数。0以上1以下。
        一般的に0.9より0.99, 0.99より0.999のほうがエラーが落ちやすい。
        """
        Network.__init__(self, input_node, layers, terminal_nodes)
        self.momentum_coeff = momentum_coeff


    def calc_grads(self, wrt):
        """
        wrt: stands for "With Respect To".
        何に対する偏微分かを指定する。
        ここで、速度項をつくっておく。
        """
        parameters = wrt
        velocities = []
        for param in parameters:
            value = numpy.zeros_like(param.get_value(), dtype='float32')
            velocities.append(theano.shared(value, param.name))
        grads = theano.grad(self.cost, wrt=wrt)
        self.grad_map = zip(wrt, grads, velocities)
        assert 0 < len(self.grad_map), (wrt, grads)


    def get_trainer(self, learning_rate, data_dict):
        """
        :param learning_rate: 学習率。焼き鈍しによる変化があるかも。

        :param data_dict: str をキーとし、
        学習用データセットを値とする。値は theano.shared()。

        :return: 速度項の更新をする関数・パラメータの更新をする関数の二つ。
        この順番で。
        使うときは、まず速度項を更新し、つぎにパラメータの更新をすること。
        二つをひとつの theano.function にすると、
        現在の更新後の速度項を使えないので、 CM と異なる。
        """
        velocity_update_symbols = OrderedDict()
        param_update_symbols = OrderedDict()
        for param_i, grad_i, velocity_i in self.grad_map:
            # 勾配が nan になる場合がある。そのときはパラメータを更新しない。

            velocity_update_symbols[velocity_i] = tensor.switch(
                tensor.isnan(grad_i),
                velocity_i,
                self.momentum_coeff * velocity_i - learning_rate * grad_i
            )
            param_update_symbols[param_i] = param_i + velocity_i
        index = tensor.iscalar()
        givens = OrderedDict()
        for k, v in data_dict.items():
            assert k in self.terminal_symbols
            givens[k] = v[index : index + self.input_node.batch_size]
        velocity_update_func = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=velocity_update_symbols,
            givens=givens
        )
        param_update_func = theano.function(
            inputs=[index],
            outputs=self.cost,
            updates=param_update_symbols,
            givens=givens
        )
        return velocity_update_func, param_update_func


class DataGenerator(object):
    """
    学習・テスト用データセットを、ネットワークに渡せるデータに変換するクラス。

    :param data_nodes:
        DataNode のリスト。
        全ての data_nodes.files の要素数は等しくなければならない。
    """
    def __init__(self, *data_nodes):
        self.nodes = OrderedDict()
        for node in data_nodes:
            self.nodes[node.name] = node
        self._data_generator = self.__get_data_generator()
        self.next()

    def __get_data_generator(self):
        """
        ファイル列から無限にデータを得るジェネレータ。
        """
        rng = numpy.random.RandomState()
        orders = range(self.n_files)
        while True:
            # rng.shuffle(orders)
            for i in orders:
                result = OrderedDict()
                for node in self.nodes.values():
                    assert len(node.files) == len(orders)
                    node.update(i)
                    result[node.symbol] = node.shared
                yield result

    @property
    def n_data(self):
        """
        1ファイル中のデータ数。どのファイルも一定という前提。
        """
        return self.nodes.values()[0].n_data

    @property
    def n_files(self):
        """
        ファイル数。どのノードも一定のファイル数という前提。
        """
        return len(self.nodes.values()[0].files)

    @property
    def ndarray_dict(self):
        """
        theano.Variable をキーとし theano.shared() を値とする
        辞書を返すのではなく、 str をキーとし numpy.ndarray を値とする
        辞書を返す。
        """
        shared_dict = self._data
        return {
            k.name:v.get_value(borrow=True)
            for k, v in shared_dict.items()
        }

    def next(self):
        self._data = self._data_generator.next()
        return self._data

