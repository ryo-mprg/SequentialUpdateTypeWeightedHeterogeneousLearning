# -*- coding: UTF-8 -*-
"""
layer
================================

architecture(networks) を構成する各層のクラス

全てのクラスは、 _Base のサブクラスである。

入出力のデータ形状の変わらない層は、
_Activator のサブクラスとして定義する。

入力のノードのほかに入力したいノードがあるときは、 _2in1 のサブクラスとして
定義する。

学習によるチューニングするパラメータを持つクラスは、 _Parametric のサブクラス
として定義する。
"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import logging
from os import path

import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.tensor.nnet.conv import ConvOp

from node import Node
from _piece import floatX
from _piece import Parameter


class _Base(object):
    """
    レイヤーの基本クラス。
    """
    def chain(self, node, weights):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def __str__(self):
        class_name = self.__class__.__name__
        return '{0}: {1}'.format(class_name, self.detail_str())

    def detail_str(self):
        return ''


class _Activator(_Base):
    """
    活性化関数。基本クラス。 Sigmoid, Tanh など。
    要素ごとに計算するので、次元は変化しない。
    このクラス単体でも、レイヤーを構成してよい。

    _Activator のサブクラスとして定義されたクラスのインスタンスは、
    _Parametric クラスの活性化関数として利用することができる。
    """
    def chain_node(self, node, weights):
        'サブクラスは、 self.act のみ定義すれば良い。'
        return Node(node.shape, self.act(node.symbol))


class _2in1(_Base):
    """
    2つのノードを1つのノードにするクラス。基本クラス。
    """
    def __init__(self, layer2):
        """
        初期化時に他方のノードを指定する。

        :param layer2:
            二番目のノード。 chain_node() では一つのノードしか
            引数に取れないので、ここでもう一つのノードを指定しておく。
        """
        self.layer2 = layer2


class _Parametric(_Base):
    """
    パラメータを持つレイヤー。基本クラス。
    """
    def __init__(self, tune=False, **kwargs):
        """
        :param tune:
            真ならば、学習によってパラメータ更新をする。
        """
        self._tune = tune
        if 'prefix' in kwargs:
            self._prefix = kwargs['prefix']

    def set_prefix(self, sn):
        """
        パラメータの保存先ファイルの接頭辞。
        prefix + '_weight.pkl' と prefix + '_bias.pkl' というファイルに
        対して、パラメータ読み込みとパラメータ保存が行われる。

        prefix には、フォルダ名は含まれない。フォルダ名は load()メソッド、
        save() メソッドの引数に与えるから。

        デフォルトでは、ファイルの入出力を行わない。
        """
        if not hasattr(self, '_prefix'):
            self._prefix = '{:02d}_{:.4s}'.format(sn, self.__class__.__name__)

    def build_param(self, **kwargs):
        """
        ゼロクリアされたパラメータを構築する。
        この関数が呼ばれてはじめて、 weight, bias がつくられる。

        :param weight_shape: weight の形状
        :param bias_shape: bias の形状
        """
        self.params = {}
        for k in kwargs:
            self.params[k] = Parameter(kwargs[k], k)
        # 初期化時に filename の設定をしたいところだが、
        # chain() が呼ばれた後でないとパラメータの shape が決まらないので、
        # このタイミングで filename の設定をする。
        if hasattr(self, '_prefix'):
            for v in self.params.values():
                v.build_filename(self._prefix)

    def init_param(self, rng):
        """
        パラメータの初期化。サブクラスに実装を移譲。
        """
        raise NotImplementedError

    @property
    def tune(self):
        """
        :return: 真ならば、パラメータ更新をする
        """
        return self._tune

    def load(self, result_folder):
        """
        パラメータをファイルから読む。
        初期化時に prefix 設定されていなければ、何もしない。

        :param result_folder: 保存先のフォルダ名
        """
        for v in self.params.values():
            v.load(result_folder)

    def save(self, result_folder):
        """
        パラメータをファイルに書く。
        初期化時に prefix 設定されていなければ、何もしない。

        :param result_folder: 保存先のフォルダ名
        """
        for v in self.params.values():
            v.save(result_folder)

    def push(self):
        """
        現在のパラメータを一時的に保存する。
        保存したパラメータに戻したいときには、 pop() を使う。
        """
        for v in self.params.values():
            v.push()

    def pop(self):
        """
        push() で保存したパラメータに戻す。
        """
        for v in self.params.values():
            v.pop()


class Linear(_Activator):
    'ax + b'
    def __init__(self, multiplier=1.0, accumulator=0.0):
        """
        初期化時に係数を設定する。デフォルト(引数なし)の場合は、
        y = x になる。

        :param multiplier: かける数(a)。デフォルトは 1
        :param accumulator: たす数(b)。デフォルトは 0
        """
        self.multiplier = multiplier
        self.accumulator = accumulator
        self.act = lambda x: multiplier * x + accumulator


class Plain(_Activator):
    """
    通すだけの層。
    Convolution のあとに MaxOut を置いた場合は、十分に非線形性が得られるためか、
    Convolution の活性化関数に、複雑な Sigmoid などを入れる必要がなく、
    Plain で十分。
    """
    def act(self, x):
        return x


class Tanh(_Activator):
    """
    tanh。プレトレーニングなしなら、 Convlutional 層の活性化関数として最適。
    """
    def __init__(self):
        self.act = tensor.tanh


class Sigmoid(_Activator):
    """
    sigmoid関数。
    プレトレーニングありなら、 Convlutional 層の活性化関数として有効。
    """
    def __init__(self):
        self.act = tensor.nnet.sigmoid


class Sgn(_Activator):
    """
    [-1, 1] -> {-1, 1}
    要するに、符号を返す。ノードごとに二値化する、とも言える。
    """
    def __init__(self):
        self.act = tensor.sgn


class ReLU(_Activator):
    """
    Rectified Linear Unit

    Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey.
    ImageNet classification with deep convolutional neural
    networks. In Advances in Neural Information Processing
    Systems 25 (NIPS’2012). 2012.

    http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    """
    def act(self, x):
        return tensor.maximum(x, 0)


class SoftMax(_Activator):
    """
    softmax = exp(x) / Σexp(x)

    マルチクラス識別の学習で、 NegLog() の直前にこれを入れる。
    形状は変化しない。
    """
    def __init__(self):
        self.act = tensor.nnet.softmax


class Logmoid(_Activator):
    """
    log()を使った Sigmoid っぽい関数。
    logmoid(x) = sgn(x) * log(abs(x) + 1)

    微分はこうなって、結構きれい。
    logmoid(x)' = 1 / (sgn(x) + 1)

    CIFAR-10 で試したとき、 Tanh より若干良かった。
    """
    def __init__(self):
        def _logmoid(x):
            return tensor.sgn(x) * tensor.log(abs(x) + 1)
        self.act = _logmoid


class Normalize(_Activator):
    """
    サンプルごとにノードの値を平均0分散1にする。
    """
    def act(self, x):
        shape = x.shape
        x = x.flatten(2)
        mean = x.mean(axis=1, keepdims=True)
        stddev = x.std(axis=1, keepdims=True)
        x = tensor.switch(tensor.lt(0, stddev), (x - mean) / stddev, x)
##        x = (x - mean) / stddev
        return x.reshape(shape)


class SubtractiveNormalize(_Activator):
    """
    サンプルごとにノードの値を平均0にする。
    """
    def act(self, x):
        shape = x.shape
        x = x.flatten(2)
        mean = x.mean(axis=1, keepdims=True)
        x -= mean
        return x.reshape(shape)


class MinMaxNormalize(_Activator):
    """
    サンプルごとに最小値0最大値1にする。
    """
    def act(self, x):
        min_x = x.min()
        max_x = x.max()
        minmax = max_x - min_x
        x = tensor.switch(tensor.lt(0, minmax), (x - min_x) / minmax, x)
        return x

class DropOut(_Activator):
    """
    Drop out

    Hinton, Geoffrey E., Srivastava, Nitish, Krizhevsky, Alex,
    Sutskever, Ilya, and Salakhutdinov, Ruslan. Improving
    neural networks by preventing co-adaptation of feature
    detectors. Technical report, arXiv:1207.0580, 2012.

    http://arxiv.org/pdf/1207.0580.pdf

    Full() の入力直前に学習時は DropOut(0.5) を入れ、
    テスト・予測時は Linear(0.5, 0) を入れると、汎化性能があがる。

    なお、 denoising auto encoder のときにも、この層を使う。
    denoising auto encoder の場合は、学習したいlayer の入力直前に
    DropOut(0.2) くらいを入れると効果的。drop_rate が大き過ぎると
    きちんと学習してくれない。
    """
    def __init__(self, drop_rate):
        """
        drop_rate: 入力ノードを落として、値を0にする確率。
        入力ノードが残る確率は、 1 - drop_rate。
        """
        self.pass_rate = 1.0 - drop_rate

    def act(self, x):
        rng = RandomStreams()  # コードの簡素化のため、シードの制御をあきらめた
        mask = rng.binomial(size=x.shape, p=self.pass_rate, dtype=floatX)
        return mask * x


class SwapNode(_Activator):
    """
    ノードの位置をランダムで入れ替える
    人の曖昧性を反映させたり微妙なずれへの対応
    """

    def __init__(self, swap_rate):
        """
        :param swap_rate: ノードの入れ替わる確率。
        (0, 0.5) の範囲内であること。これは実装上の制約。
        """
        self.border = self.rate2border(swap_rate)

    def rate2border(self, swap_rate):
        """
        スワップ率 XX にしてとユーザから指定されたときに、
        内部パラメータ(ノイズの振幅)を決める。

        この関数はやたらと長いが、そのほとんどは参照テーブル。
        参照テーブルは、事前に実施した実験によって求めた。

        :param swap_rate: ユーザが初期化時に指定するスワップ率。
        0以上0.5未満である、という前提。
        """
        class Rate2Border:
            """
            下限(low, inclusive), 上限(high, exclusive), 内部パラメータ(border)
            を保持するクラス。スワップ率 XX にしてとユーザから指定されたときに、
            内部パラメータを決めるために使う。
            """
            def __init__(self, low, high, border):
                self.low = low
                self.high = high
                self.border = border
            def is_in_range(self, rate):
                return self.low < rate < self.high

        # 参照テーブル
        RATE2BORDER = (
            Rate2Border(0.0, 0.000396, 0.5),
            Rate2Border(0.000396, 0.001406, 0.51),
            Rate2Border(0.001406, 0.003236, 0.52),
            Rate2Border(0.003236, 0.005474, 0.53),
            Rate2Border(0.005474, 0.0084, 0.54),
            Rate2Border(0.0084, 0.011184, 0.55),
            Rate2Border(0.011184, 0.01497, 0.56),
            Rate2Border(0.01497, 0.01879, 0.57),
            Rate2Border(0.01879, 0.023506, 0.58),
            Rate2Border(0.023506, 0.028018, 0.59),
            Rate2Border(0.028018, 0.032664, 0.6),
            Rate2Border(0.032664, 0.037528, 0.61),
            Rate2Border(0.037528, 0.042728, 0.62),
            Rate2Border(0.042728, 0.047774, 0.63),
            Rate2Border(0.047774, 0.052784, 0.64),
            Rate2Border(0.052784, 0.059072, 0.65),
            Rate2Border(0.059072, 0.063976, 0.66),
            Rate2Border(0.063976, 0.06963, 0.67),
            Rate2Border(0.06963, 0.075356, 0.68),
            Rate2Border(0.075356, 0.080956, 0.69),
            Rate2Border(0.080956, 0.087584, 0.7),
            Rate2Border(0.087584, 0.094006, 0.71),
            Rate2Border(0.094006, 0.098908, 0.72),
            Rate2Border(0.098908, 0.105084, 0.73),
            Rate2Border(0.105084, 0.110802, 0.74),
            Rate2Border(0.110802, 0.117382, 0.75),
            Rate2Border(0.117382, 0.12283, 0.76),
            Rate2Border(0.12283, 0.129128, 0.77),
            Rate2Border(0.129128, 0.135424, 0.78),
            Rate2Border(0.135424, 0.14147, 0.79),
            Rate2Border(0.14147, 0.145944, 0.8),
            Rate2Border(0.145944, 0.153334, 0.81),
            Rate2Border(0.153334, 0.157306, 0.82),
            Rate2Border(0.157306, 0.163382, 0.83),
            Rate2Border(0.163382, 0.169608, 0.84),
            Rate2Border(0.169608, 0.175334, 0.85),
            Rate2Border(0.175334, 0.180998, 0.86),
            Rate2Border(0.180998, 0.18586, 0.87),
            Rate2Border(0.18586, 0.19269, 0.88),
            Rate2Border(0.19269, 0.197142, 0.89),
            Rate2Border(0.197142, 0.202794, 0.9),
            Rate2Border(0.202794, 0.208326, 0.91),
            Rate2Border(0.208326, 0.213902, 0.92),
            Rate2Border(0.213902, 0.219378, 0.93),
            Rate2Border(0.219378, 0.224048, 0.94),
            Rate2Border(0.224048, 0.229176, 0.95),
            Rate2Border(0.229176, 0.234928, 0.96),
            Rate2Border(0.234928, 0.239708, 0.97),
            Rate2Border(0.239708, 0.245732, 0.98),
            Rate2Border(0.245732, 0.249756, 0.99),
            Rate2Border(0.249756, 0.255384, 1.0),
            Rate2Border(0.255384, 0.260103, 1.01),
            Rate2Border(0.260103, 0.264787, 1.02),
            Rate2Border(0.264787, 0.270572, 1.03),
            Rate2Border(0.270572, 0.274675, 1.04),
            Rate2Border(0.274675, 0.280994, 1.05),
            Rate2Border(0.280994, 0.286997, 1.06),
            Rate2Border(0.286997, 0.290517, 1.07),
            Rate2Border(0.290517, 0.294761, 1.08),
            Rate2Border(0.294761, 0.301874, 1.09),
            Rate2Border(0.301874, 0.305469, 1.1),
            Rate2Border(0.305469, 0.31164, 1.11),
            Rate2Border(0.31164, 0.316401, 1.12),
            Rate2Border(0.316401, 0.321315, 1.13),
            Rate2Border(0.321315, 0.327481, 1.14),
            Rate2Border(0.327481, 0.331819, 1.15),
            Rate2Border(0.331819, 0.337332, 1.16),
            Rate2Border(0.337332, 0.342045, 1.17),
            Rate2Border(0.342045, 0.347259, 1.18),
            Rate2Border(0.347259, 0.352315, 1.19),
            Rate2Border(0.352315, 0.356092, 1.2),
            Rate2Border(0.356092, 0.361195, 1.21),
            Rate2Border(0.361195, 0.366743, 1.22),
            Rate2Border(0.366743, 0.370591, 1.23),
            Rate2Border(0.370591, 0.375608, 1.24),
            Rate2Border(0.375608, 0.381261, 1.25),
            Rate2Border(0.381261, 0.386622, 1.26),
            Rate2Border(0.386622, 0.389732, 1.27),
            Rate2Border(0.389732, 0.394616, 1.28),
            Rate2Border(0.394616, 0.399776, 1.29),
            Rate2Border(0.399776, 0.403685, 1.3),
            Rate2Border(0.403685, 0.408463, 1.31),
            Rate2Border(0.408463, 0.412025, 1.32),
            Rate2Border(0.412025, 0.416646, 1.33),
            Rate2Border(0.416646, 0.421323, 1.34),
            Rate2Border(0.421323, 0.427014, 1.35),
            Rate2Border(0.427014, 0.431291, 1.36),
            Rate2Border(0.431291, 0.433513, 1.37),
            Rate2Border(0.433513, 0.439265, 1.38),
            Rate2Border(0.439265, 0.44276, 1.39),
            Rate2Border(0.44276, 0.445605, 1.4),
            Rate2Border(0.445605, 0.450644, 1.41),
            Rate2Border(0.450644, 0.45417, 1.42),
            Rate2Border(0.45417, 0.45892, 1.43),
            Rate2Border(0.45892, 0.462962, 1.44),
            Rate2Border(0.462962, 0.465763, 1.45),
            Rate2Border(0.465763, 0.469157, 1.46),
            Rate2Border(0.469157, 0.474, 1.47),
            Rate2Border(0.474, 0.476811, 1.48),
            Rate2Border(0.476811, 0.480656, 1.49),
            Rate2Border(0.480656, 0.484658, 1.5),
            Rate2Border(0.484658, 0.489279, 1.51),
            Rate2Border(0.489279, 0.492696, 1.52),
            Rate2Border(0.492696, 0.496414, 1.53),
            Rate2Border(0.496414, 0.499358, 1.54),
            Rate2Border(0.499358, 0.503531, 1.55),
        )
        assert 0 <= swap_rate < 0.503531, swap_rate
        for d in RATE2BORDER:
            if d.is_in_range(swap_rate):
                return d.border
        raise ValueError # ここに来たらだめ

    def act(self, x):
        xx = x.flatten(2)
        shape = xx.shape[1]
        rng = RandomStreams()  # コードの簡素化のため、シードの制御をあきらめた
        noise = rng.uniform(low=-self.border, high=self.border, size=(shape,))
        order = tensor.arange(shape)
        order2 = tensor.argsort(noise + order)
        symbol = xx[:, order2].reshape(x.shape)
        return symbol


class Cast(_Activator):
    """
    dtype にキャストする。
    """
    def __init__(self, dtype):
        self.act = lambda x: tensor.cast(x, dtype)


class Reshape(_Base):
    """
    ノードの形状を変形させる。
    総数(Node.size)は変化しない。
    """
    def __init__(self, *shape_without_batch):
        """
        shape_without_batch: バッチサイズを除く、変形後の形状。
        """
        self.shape_without_batch = shape_without_batch

    def chain_node(self, node, weights):
        expected = numpy.prod(self.shape_without_batch) * node.batch_size
        assert node.size == expected, 'node:{} expect:{}'.format(node, expected)
        out_shape = [node.batch_size] + list(self.shape_without_batch)
        return Node(out_shape, node.symbol.reshape(out_shape))


class Transpose(_Base):
    """
    axis の順を入れ替える

    numpy.ndarray.transpose(), theano.tensor.dimshuffle() と同様。
    """
    def __init__(self, *order):
        """
        :param order:
            バッチサイズを除く、入れ替え順。
            バッチサイズはつねに第1軸(axis=0)。

            1, 2, ..., len(order) の数値のみを使うこと。

            たとえば Transpose(2, 3, 1) だと、
            Node.shape = [10, 20, 30, 40] が [10, 30, 40, 20] になる。
        """
        assert all(i + 1 in order for i in xrange(len(order))), order
        self.order = [0] + [i for i in order]

    def chain_node(self, node, weights):
        assert len(self.order) == node.ndim, node
        shape = [node.shape[i] for i in self.order]
        symbol = node.symbol.dimshuffle(*self.order)
        return Node(shape, symbol)


class GetRange(_Base):
    """
    ノードの一部だけ取り出す
    """
    def __init__(self, start, stop, step = 1):
        self.start = start
        self.stop = stop
        self.step = step
        assert start < stop, (start, stop)
        assert 0 < step, step
        self.out_node = (stop - start) / step

    def chain_node(self, node, weights):
        assert 2 == node.ndim, node
        shape = node.shape[0], (self.stop - self.start) / self.step
        symbol = node.symbol[:, self.start:self.stop:self.step]
        return Node(shape, symbol)


class Crop(_Base):
    """
    指定サイズになるように真ん中を切り出す
    """
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def chain_symbol(self, node):
        batch_size, n_in_maps, n_in_rows, n_in_cols = node.shape
        assert self.n_rows <= n_in_rows, node
        assert self.n_cols <= n_in_cols, node
        max_row0 = n_in_rows - self.n_rows
        max_col0 = n_in_cols - self.n_cols
        row_from = max_row0 / 2
        row_to = row_from + self.n_rows
        col_from = max_col0 / 2
        col_to = col_from + self.n_cols
        return node.symbol[:, :, row_from:row_to, col_from:col_to]

    def chain_node(self, node, weights):
        assert 4 == node.ndim, node
        batch_size, n_in_maps, _, _ = node.shape
        shape = batch_size, n_in_maps, self.n_rows, self.n_cols
        symbol = self.chain_symbol(node)
        return Node(shape, symbol)


class RandomCrop(Crop):
    """
    指定サイズになるようにランダムに切り出す
    """
    def chain_symbol(self, node):
        """
        サンプルごとに切り出し位置を変えるが、
        マップについては、共通。
        たとえば、(2,3,5,7) の入力バッチについて、必要な乱数は、
        切り出し開始位置として、 2(バッチサイズ) * 2(rowとcol) = 4 個。
        """
        batch_size, n_in_maps, n_in_rows, n_in_cols = node.shape
        max_row0 = n_in_rows - self.n_rows
        max_col0 = n_in_cols - self.n_cols
        rng = RandomStreams()
        starts = rng.uniform(size=(batch_size, 2))
        starts = tensor.cast(starts * [[max_row0], [max_col0]], 'int32')
        symbols = []
        for i in xrange(batch_size):
            row_from = starts[i, 0]
            col_from = starts[i, 1]
            row_to = row_from + self.n_rows
            col_to = col_from + self.n_cols
            symbols.append(node.symbol[i, :, row_from:row_to, col_from:col_to])
        return tensor.stack(*symbols)


class Flatten(_Base):
    """
    Convolution から Full に接続するために、
    (バッチサイズ・マップ数・行数・列数) というノードの構造を、
    (バッチサイズ・ノード数) に変形する。
    """
    def chain_node(self, node, weights):
        out_shape = node.batch_size, node.size / node.batch_size
        symbol = node.symbol.flatten(2)
        return Node(out_shape, symbol)


class MaxPool(_Base):
    """
    Max pooling

    Scherer, D., Müller, A., Behnke, S.: Evaluation of pooling operations
    in convolutional architectures for object recognition. In: International
    Conference on Artificial Neural Networks (2010)

    http://www.ais.uni-bonn.de/papers/icann2010_maxpool.pdf
    """
    def __init__(self, *downscale_factors):
        """
        行・列をそれぞれ何分の一にするか、指定する。

        :param downscale_factors:
            (rows, cols)

            rows: 行数を 1/rows にする

            cols: 列数を 1/cols にする
        """
        assert 2 == len(downscale_factors)
        self.downscale_factors = downscale_factors

    def chain_node(self, node, weights):
        assert 4 == node.ndim, node
        assert node.shape[2] % self.downscale_factors[0] == 0, node
        assert node.shape[3] % self.downscale_factors[1] == 0, node
        out_shape = list(node.shape)
        out_shape[2] /= self.downscale_factors[0]
        out_shape[3] /= self.downscale_factors[1]
        symbol = max_pool_2d(node.symbol, self.downscale_factors, True)
        return Node(out_shape, symbol)

    def detail_str(self):
        return 'downscale:{}'.format(self.downscale_factors)


class AveragePool(_Base):
    """
    average pooling
    """
    def __init__(self, *downscale_factors):
        """
        行・列をそれぞれ何分の一にするか、指定する。

        :param downscale_factors:
            (rows, cols)

            rows: 行数を 1/rows にする

            cols: 列数を 1/cols にする
        """
        assert 2 == len(downscale_factors)
        self.downscale_factors = downscale_factors
        self.weight = numpy.ones(downscale_factors, dtype=floatX)
        self.weight /= self.weight.size

    def chain_node(self, node, weights):
        assert 4 == node.ndim, node
        assert node.shape[2] % self.downscale_factors[0] == 0, node
        assert node.shape[3] % self.downscale_factors[1] == 0, node
        out_shape = list(node.shape)
        out_shape[2] /= self.downscale_factors[0]
        out_shape[3] /= self.downscale_factors[1]
        zeros = numpy.zeros_like(self.weight, dtype=floatX)
        maps = node.shape[1]

        weight = []
        for r in xrange(maps):
            row_value = []
            for c in xrange(maps):
                if r == c:
                    row_value.append(self.weight)
                else:
                    row_value.append(zeros)
            weight.append(row_value)
        weight = numpy.asarray(weight, dtype=floatX)
        weight_symbol = theano.shared(value=weight, name='w', borrow=True)
        unit = conv2d(
            input=node.symbol,
            filters=weight_symbol ,
            filter_shape=weight.shape,
            image_shape=node.shape,
            border_mode='valid',
            subsample=self.downscale_factors,
            )
        return Node(out_shape, unit)


class LpPool(_Base):
    """
    Lp pooling。
    (Σx^p)^(1/p)
    """
    def __init__(self, p, *downscale_factors):
        """
        :param p:
            Lp pooling の"p"。p乗してたたみこんで1/p乗する。

        :param downscale_factors:
            (rows, cols)
            rows: 行数を 1/rows にする
            cols: 列数を 1/cols にする
        """
        assert 2 == len(downscale_factors)
        self.p = p
        self.downscale_factors = downscale_factors
        self.weight = numpy.ones(downscale_factors, dtype=floatX)

    def chain_node(self, node, weights):
        assert 4 == node.ndim, node
        assert node.shape[2] % self.downscale_factors[0] == 0, node
        assert node.shape[3] % self.downscale_factors[1] == 0, node
        out_shape = list(node.shape)
        out_shape[2] /= self.downscale_factors[0]
        out_shape[3] /= self.downscale_factors[1]
        zeros = numpy.zeros_like(self.weight, dtype=floatX)
        maps = node.shape[1]

        weight = []
        for r in xrange(maps):
            row_value = []
            for c in xrange(maps):
                if r == c:
                    row_value.append(self.weight)
                else:
                    row_value.append(zeros)
            weight.append(row_value)
        weight = numpy.asarray(weight, dtype=floatX)
        weight_symbol = theano.shared(value=weight, name='w', borrow=True)
        unit = conv2d(
            input=abs(node.symbol ** self.p),
            filters=weight_symbol ,
            filter_shape=weight.shape,
            image_shape=node.shape,
            border_mode='valid',
            subsample=self.downscale_factors,
            )
        unit = unit ** (1.0/self.p)
        return Node(out_shape, unit)


class MaxOut(_Base):
    """
    Max Out。

    Goodfellow, Ian J., et al.
    "Maxout networks." arXiv preprint arXiv:1302.4389 (2013).

    http://arxiv.org/pdf/1302.4389.pdf

    この Max Out は Convolution だけに対応。
    """
    def __init__(self, size):
        """
        size: マップ数を 1/size にする。
        """
        self.size = size

    def __verify_node(self, node):
        assert 4 == node.ndim or 2 == node.ndim, node
        assert node.shape[1] % self.size == 0, node

    def calc_out_shape(self, node):
        out_shape = list(node.shape)
        out_shape[1] /= self.size
        return out_shape

    def calc_conv_shape(self, out_shape):
        return out_shape[0], self.size, out_shape[1], 1

    def calc_core_symbol(self, symbol):
        symbol = symbol.dimshuffle(0, 3, 2, 1)
        symbol = max_pool_2d(symbol, (1, self.size), True)
        return symbol.dimshuffle(0, 3, 2, 1)

    def chain_node(self, node, weights):
        self.__verify_node(node)
        out_shape = self.calc_out_shape(node)
        symbol = node.symbol
        if 2 == node.ndim:
            conv_shape = self.calc_conv_shape(out_shape)
            symbol = symbol.reshape(conv_shape)
        symbol = self.calc_core_symbol(symbol)
        if 2 == node.ndim:
            symbol = symbol.reshape(out_shape)
        return Node(out_shape, symbol)

    def detail_str(self):
        return 'downscale:{}'.format(self.size)


class PlusMinusMaxOut(MaxOut):
    """
    Max Out の亜種。
    max(n1, n2), max(-n1, -n2)

    負の強い反応も拾う。
    """
    def calc_out_shape(self, node):
        out_shape = list(node.shape)
        out_shape[1] /= self.size
        out_shape[1] *= 2
        return out_shape

    def calc_conv_shape(self, out_shape):
        return out_shape[0], self.size, out_shape[1] / 2, 1

    def calc_core_symbol(self, symbol):
        shuffled = symbol.dimshuffle(0, 3, 2, 1)
        plus_maxout = max_pool_2d(shuffled, (1, self.size), True)
        plus_maxout = plus_maxout.dimshuffle(0, 3, 2, 1)
        minus_maxout = max_pool_2d(-shuffled, (1, self.size), True)
        minus_maxout = minus_maxout.dimshuffle(0, 3, 2, 1)
        return tensor.concatenate((plus_maxout, minus_maxout), axis=1)

    def detail_str(self):
        return 'downscale:{} / 2'.format(self.size)


class ArgMax(_Base):
    """
    argmax によるサンプルごとのラベルの推定
    top layer用だが、y と比較しないので、学習はできない。
    予測用。
    """
    def chain_node(self, node, weights):
        assert 1 < node.ndim, node
        axis = node.ndim - 1
        return Node(node.shape[:-1], node.symbol.argmax(axis=axis))


class GetLabelMap(_Base):
    """
    LabelMap() の学習結果で推定をするときに使う。ラベル画像を得る。

    事前に Reshape して、 画像次元 + ラベル次元になっていないといけない。
    たとえば、サイズ 100x100, ラベル数8 のラベル画像を得るには、直前に
    Reshape(100, 100, 8)
    を入れること。

    なお、GetLabelMap() 後の Reshape は不要。上記の例では、出力が
    (100, 100) になるので、そのままラベル画像として使える。
    """
    def chain_node(self, node, weights):
        assert 2 < node.shape, node
        out_shape = node.shape[:-1]
        tmp_shape = numpy.prod(out_shape), node.shape[-1]
        symbol = node.symbol.reshape(tmp_shape)
        symbol = tensor.nnet.softmax(symbol)
        symbol = tensor.argmax(symbol, -1)
        symbol = symbol.reshape(out_shape)
        return Node(out_shape, symbol)


class Concatenate(_2in1):
    """
    二つのノードを連結するクラス。
    """
    def chain_node(self, node, weights):
        assert 2 == node.ndim, node
        node2 = self.layer2.node
        assert node2.ndim == node.ndim, node
        batch_size, rest = node.shape
        _, rest_b = node2.shape
        out_shape = batch_size, (rest + rest_b)
        symbol = tensor.concatenate([node.symbol, node2.symbol], axis=1)
        return Node(out_shape, symbol)


class NegativeLogLikelihood(_2in1):
    """
    負の対数尤度。
    Logistic Regression の頭になる。
    マルチクラス判別などで使われる。
    """
    def chain_node(self, node, weights):
        assert node.ndim == 2, node
        node2 = self.layer2.node
        assert node2.ndim == 1, node2
        symbol = -tensor.log(node.symbol)
        symbol = tensor.switch(tensor.isnan(symbol), 0, symbol)
        symbol = symbol[
            tensor.arange(node.batch_size),
            node2.symbol
        ]
        return Node((node.batch_size,), symbol)


class MeanSquaredError_facial(_2in1):
    """
    Mean Squared Error。L2ノルム二乗誤差。
    二値画像を学習するときに使う top layer。
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = ((node.symbol - node2.symbol) ** 2) * weights
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class MeanSquaredError_age(_2in1):
    """
    Mean Squared Error。L2ノルム二乗誤差。
    二値画像を学習するときに使う top layer。
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = ((node.symbol - node2.symbol) ** 2) * weights
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class MeanSquaredError_smile(_2in1):                                                                                       
    """
    Mean Squared Error。L2ノルム二乗誤差。
    二値画像を学習するときに使う top layer。
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = ((node.symbol - node2.symbol) ** 2) * weights
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class MeanSquaredError_class_gender(_2in1):
    """
    Mean Squared Error。L2ノルム二乗誤差。
    二値画像を学習するときに使う top layer。
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = ((node.symbol - node2.symbol) ** 2) * weights
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class MeanSquaredError_class_race(_2in1):
    """
    Mean Squared Error。L2ノルム二乗誤差。
    二値画像を学習するときに使う top layer。
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = ((node.symbol - node2.symbol) ** 2) * weights
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class BinaryCrossEntropy(_2in1):
    """
    二値画像を学習するときに使う top layer。
    """
    def chain_node(self, node, weights):
        """
        node: 予測値。[0, 1]の連続値と想定。
        node2: 真値。ほぼ{0, 1}の離散値と想定。
        """
        node2 = self.layer2.node
        assert node.shape == node2.shape, (str(node), str(node2))
        symbol = tensor.nnet.binary_crossentropy(
            node.symbol,
            node2.symbol
        )
        symbol = symbol.flatten(2).mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class LabelMap(_2in1):
    """
    出力は各画素が離散値を取る画像。ラベル画像とか。
    """
    def chain_node(self, node, weights):
        """
        :param node: 予測値。最後の次元が、各ラベルの尤度(確率)。
        """
        node2 = self.layer2.node
        assert node.ndim == node2.ndim + 1, (node.shape, node2.shape)
        assert node2.shape == node.shape[:-1]
        tmp_shape = node2.size, node.shape[-1]
        symbol = node.symbol.reshape(tmp_shape)
        symbol = tensor.nnet.softmax(symbol)
        symbol = -tensor.log(symbol)
        symbol = symbol[
            tensor.arange(node2.size),
            tensor.cast(node2.symbol, 'int32').flatten(1)
        ]
        symbol = symbol.reshape(node2.shape).flatten(2)
        symbol = symbol.mean(axis=1)
        # mean にしておかないと、 NaN が出やすい。
        return Node((node.batch_size,), symbol)


class LabelMapNegativeLog(_2in1):
    """
    教師信号と同じ形状のまま、 負の対数尤度を得る。

    #242(LabelMap のクラスごとの誤差) の対応のため。
    LabelMap ごとの誤差を theano のシンボリック表現を求めようとしたら、
    マスク画像を作るか、頻度分布みたいなのをつくるかしないといけない。
    データ生成を theano のシンボリック表現でつかうのは複雑なので、
    最後のクラス毎の誤差の計算は theano.function の外側でやってもらう。
    外側、というのは、 calc_error() のこと。
    """
    def chain_node(self, node, weights):
        """
        ここでは、教師信号と同じ形状の、負の対数尤度を出すまで。

        :param node: 予測値。最後の次元が、各ラベルの尤度(確率)。
        """
        node2 = self.layer2.node
        assert node.ndim == node2.ndim + 1, (node.shape, node2.shape)
        assert node2.shape == node.shape[:-1]
        tmp_shape = node2.size, node.shape[-1]
        symbol = node.symbol.reshape(tmp_shape)
        symbol = tensor.nnet.softmax(symbol)
        symbol = -tensor.log(symbol)
        symbol = symbol[
            tensor.arange(node2.size),
            tensor.cast(node2.symbol, 'int32').flatten(1)
        ]
        symbol = symbol.reshape(node2.shape)
        return Node(node2.shape, symbol)

    def calc_error(self, predict, expect):
        """
        クラスごとのエラー
        """
        assert predict.shape == expect.shape
        n_classes = numpy.ptp(expect.astype(int)) + 1
        # 1から始めるのは0割防止
        error_for_each_class = numpy.ones((n_classes,), dtype=float)
        count_for_each_class = numpy.ones((n_classes,), dtype=int)
        for p, e in zip(predict.flatten(), expect.flatten()):
            error_for_each_class[e] += p
            count_for_each_class[e] += 1
        logging.info('count_for_each_class: {}'.format(count_for_each_class))
        return error_for_each_class / count_for_each_class


class Unpool(_2in1):
    """
    Unpooling

    Visualizing and Understanding Convolutional Networks
    M.D. Zeiler, R. Fergus
    Arxiv 1311.2901 (Nov 28, 2013)

    http://arxiv.org/pdf/1311.2901v3

    max pooling の逆。
    max pooling 前のノードを入力とする。

    出力は max pooling 前と同じ形状。
    local maxima の位置だけ伝播し、他は0にする。

    max を得るブロックは分かれていること。
    (ブロックサイズ * ブロック数 == マップサイズ)

    FIXME: 未完成
    """
    def chain_node(self, node, weights):
        node2 = self.layer2.node
        assert 4 == node.ndim, node
        out_shape = node2.shape
        assert 0 == out_shape[-2] % node.shape[-2], (out_shape, node.shape)
        assert 0 == out_shape[-1] % node.shape[-1], (out_shape, node.shape)
        coeff_row = int(out_shape[-2]) / int(node.shape[-2])
        coeff_col = int(out_shape[-1]) / int(node.shape[-1])
##        tmp_shape = node.shape[0], node.shape[1], node.shape[2], out_shape[3]
##        symbol = node.symbol.repeat(coeff_col, axis=-1).reshape(tmp_shape)
##        symbol = symbol.repeat(coeff_row, axis=-2).reshape(out_shape)
##        symbol_b = node2.symbol
##        symbol = tensor.switch(tensor.lt(abs(symbol - symbol_b), 1e-4), symbol_b, 0)
        symbol = node.symbol.repeat(coeff_col, axis=-1)
        symbol = symbol.repeat(coeff_row, axis=-2)
        symbol_b = node2.symbol
        symbol = tensor.switch(tensor.lt(abs(symbol - symbol_b), 1e-3), symbol_b, 0)
        return Node(out_shape, symbol)


class MultiClass_gender(_Base):
    """
    複数クラスの識別
    """
    def __init__(self, out_node):
        self.out_node = out_node

    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights[1])  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        out_node = self.out_node
        layer = Cast('int32')
        out_node = layer.chain(out_node, weights)
        toplayer = MeanSquaredError_class_gender(layer)
        return toplayer.chain(node, weights)

class MultiClass_race(_Base):
    """
    複数クラスの識別
    """
    def __init__(self, out_node):
        self.out_node = out_node

    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights[3])  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        out_node = self.out_node
        layer = Cast('int32')
        out_node = layer.chain(out_node, weights)
        toplayer = MeanSquaredError_class_race(layer)
        return toplayer.chain(node, weights)

class Regression_facial(_2in1):
    """
    複数の回帰
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights[0])  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        shape_without_batch = node.shape[1:]
        layer = Reshape(*shape_without_batch)
        layer.chain(self.layer2.node, weights)
        toplayer = MeanSquaredError_facial(layer)
        return toplayer.chain(node, weights)

class Regression_age(_2in1):
    """
    複数の回帰
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights[2])  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        shape_without_batch = node.shape[1:]
        layer = Reshape(*shape_without_batch)
        layer.chain(self.layer2.node, weights)
        toplayer = MeanSquaredError_age(layer)
        return toplayer.chain(node, weights)

class Regression_smile(_2in1):
    """
    複数の回帰
    """
    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights[4])  # サブクラスに実装を移譲

        return self.node

    def chain_node(self, node, weights):
        shape_without_batch = node.shape[1:]
        layer = Reshape(*shape_without_batch)
        layer.chain(self.layer2.node, weights)
        toplayer = MeanSquaredError_smile(layer)
        return toplayer.chain(node, weights)

class Heterogeneous(_Base):
    """
    MultiClass と Regression の混合
    """
    def __init__(self, *out_layer_pairs):
        """
        :param out_layer_pairs:
            (消費するノード数, (レイヤー, レイヤー, ...))
            を要素とするリストかタプル。
        """
        self.out_layer_pairs = out_layer_pairs

    def chain_node(self, node, weights):
        in_nodes = [v[0] for v in self.out_layer_pairs]
        assert numpy.sum(in_nodes) == node.shape[1], (in_nodes, node.shape)
        start = 0
        symbol = None
        symbols = []
        for outs, layers in self.out_layer_pairs:
            end = start + outs
            in_node_shape = node.batch_size, outs
            in_node = Node(in_node_shape, node.symbol[:, start:end])
            out_node = in_node
            for l in layers:
                out_node = l.chain(out_node)
            #if symbol is None:
            #    symbol = out_node.symbol
            #else:
            #    symbol += out_node.symbol
            if out_node.ndim < 2:
                out_node.symbol = out_node.symbol.reshape((out_node.shape[0], 1))
            symbols.append(out_node.symbol)
            start = end
        symbol = tensor.concatenate(symbols, axis=1)
        return Node((node.batch_size,), symbol)


class WeightedHeterogeneous(_Base):
    """
    MultiClass と Regression の混合
    """
    def __init__(self, *out_layer_pairs):
        """
        :param out_layer_pairs:
            (消費するノード数, (レイヤー, レイヤー, ...))
            を要素とするリストかタプル。
        """
        self.out_layer_pairs = out_layer_pairs

    def chain(self, node, weights=1.0):
        """
        :param node: 入力ノード。
        :return: 出力ノード。
        """
        assert isinstance(node, Node), type(node)
        self.node = self.chain_node(node, weights)  # サブクラスに実装を移譲

        return self.node
 
    def chain_node(self, node, weights):
        in_nodes = [v[0] for v in self.out_layer_pairs]
        assert numpy.sum(in_nodes) == node.shape[1], (in_nodes, node.shape)
        start = 0
        symbol = None
        symbols = []
        for outs, layers in self.out_layer_pairs:
            end = start + outs
            in_node_shape = node.batch_size, outs#(10, 10) (10, 2) (10, 1) (10, 3) (10, 1)？
            in_node = Node(in_node_shape, node.symbol[:, start:end])
            out_node = in_node
            for l in layers:
                out_node = l.chain(out_node, weights)
            #if symbol is None:
            #    symbol = out_node.symbol
            #else:
            #    symbol += out_node.symbol
            if out_node.ndim < 2:
                out_node.symbol = out_node.symbol.reshape((out_node.shape[0], 1))
            symbols.append(out_node.symbol)
            start = end
        symbol = tensor.concatenate(symbols, axis=1)
        return Node((node.batch_size,), symbol)

class Full(_Parametric):
    """
    重み行列、バイアスをパラメータとして持つ、 fully-connected layer。
    """
    def __init__(self, outs, act, **kwargs):
        """
        :param outs:
            バッチサイズを除く出力ノード数。

            マルチクラス識別を行う場合は、 outs にクラス数を指定し、
            act に SoftMax を指定すること。また、直後の層は、
            NegativeLogLikelihood にすること。

            たとえば、 MNIST([0-9]の手書き文字認識)の識別は10クラスだから、
            階層構造の最後は次のようにする。::

                layers = (
                    ...,
                    layer.Full(10, layer.SoftMax(), ...),
                    layer.NegativeLogLikelihood(),
                )

        :param act:
            活性化関数。非線形関数。 _Activator の(サブクラスの)インスタンス。

            マルチクラス識別を行う場合は、 outs にクラス数を指定し、
            act に SoftMax を指定すること。また、直後の層は、
            NegativeLogLikelihood にすること。
        """
        _Parametric.__init__(self, **kwargs)
        assert isinstance(outs, int)
        assert isinstance(act, _Activator)
        self.each_outs = outs
        self.act = act

    def chain_node(self, node, weights):
        """
        :param node:
            shape == (バッチサイズ, ノード数) なノード。
            Convolutional 層からこのクラスに接続するときは、
            Flatten() を使う。::

                layers = (
                    ...,
                    layer.Convolution(...),
                    layer.Flatten(),
                    layer.Full(...),
                )
        """
        assert node.ndim == 2, node
        out_shape = list(node.shape)
        out_shape[-1] = self.each_outs

        weight_shape = (node.shape[-1], out_shape[-1])
        bias_shape = (out_shape[-1],)
        self.build_param(weight=weight_shape, bias=bias_shape)
        self.weight = self.params['weight']
        self.bias = self.params['bias']

        wxb = tensor.dot(node.symbol, self.weight.symbol) + self.bias.symbol
        return Node(out_shape, self.act.act(wxb))

    def init_param(self, rng):
        """
        パラメータを乱数生成器で初期化。
        _Base.chain() の後で呼ばないと、 self.weight が見つからない。

        :param rng:
            乱数生成器。 numpy.random.RandomState() のインスタンス。
        """
        bound = numpy.sqrt(6. / numpy.sum(self.weight.value.shape))
        self.weight.init_value(rng, bound)

    def detail_str(self):
        """
        詳細情報として活性化関数を表示する。
        """
        sum_param = self.weight.value.size + self.bias.value.size
        return 'act:{0}'.format(self.act)


class DropConnect(Full):
    """
    DropConnect

    Wan, Li, Zeiler, Matthew D., Zhang, Sixin, LeCun, Yann and Fergus, Rob.
    "Regularization of Neural Networks using DropConnect.."
    Paper presented at the meeting of the ICML (3), 2013.

    http://cs.nyu.edu/~wanli/dropc/dropc.pdf

    テスト(予測)のときに、単純に 0.5 をするのでない、
    何か複雑な処理を入れる必要がある。そこは未実装。

    簡易的には、 Dropout と同様に Full 層の入力直前で 全ノードを
    0.5 倍にすればいいだろう。

    なお、 DropConnect のパラメータファイル名には、 Drop という文字列が入るが
    Full のパラメータファイル名は、 Full という文字列が入る。したがって、
    DropConnect でチューニングしたパラメータを Full で使いたい場合は、
    パラメータファイル名の Drop を Full に変更する必要がある。

    変更するのが手間だと思うなら、 DropConnect を使って、 drop_rate=0.0
    を指定すればよい。
    """
    def __init__(self, outs, act, drop_rate, **kwargs):
        Full.__init__(self, outs, act, **kwargs)
        self.pass_rate = 1.0 - drop_rate

    def chain_node(self, node, weights):
        assert node.ndim == 2, node

        out_shape = list(node.shape)
        out_shape[-1] = self.each_outs

        weight_shape = (node.shape[-1], out_shape[-1])
        bias_shape = (out_shape[-1],)
        self.build_param(weight=weight_shape, bias=bias_shape)
        self.weight = self.params['weight']
        self.bias = self.params['bias']

        rng = RandomStreams()  # コードの簡素化のため、シードの制御をあきらめた
        mask = rng.binomial(
            size=self.weight.value.shape,
            p=self.pass_rate,
            dtype=floatX
        )

        wxb = tensor.dot(node.symbol, mask * self.weight.symbol) + self.bias.symbol
        return Node(out_shape, self.act.act(wxb))


class Convolution(_Parametric):
    """
    Convolutional Neural Network の Convolutional層。
    Full と異なり、入力データの次元が多くても少なくても、パラメータの次元は同じ。
    """
    _mode = 'valid'
    def __init__(self, filter_shape, act, **kwargs):
        """
        :param filter_shape:
            (filters, rows, cols)な3要素のタプル(またはリスト)。

            * filters はフィルタ数。出力する応答マップの個数。
            * rows はフィルタの行数。全フィルタ共通。
              ``node.shape[2] -= rows - 1`` になる。
            * cols はフィルタの列数。全フィルタ共通。
              ``node.shape[3] -= cols - 1`` になる。

        :param act:
            活性化関数。非線形関数。 _Activator の(サブクラスの)インスタンス。
        """
        _Parametric.__init__(self, **kwargs)
        assert 3 == len(filter_shape), filter_shape
        assert isinstance(act, _Activator)
        self.filter_shape = filter_shape
        self.act = act

    def calc_weight_shape(self, node):
        assert 4 == len(node.shape), node
        batch_size, imaps, irows, icols = node.shape
        filters, frows, fcols = self.filter_shape
        inshp = irows, icols
        kshp = frows, fcols
        orows, ocols = ConvOp.getOutputShape(inshp, kshp, mode=self._mode)
        assert 0 < orows, orows
        assert 0 < ocols, ocols
        self.each_outs = filters, orows, ocols
        out_shape = batch_size, filters, orows, ocols
        weight_shape = filters, imaps, frows, fcols
        return out_shape, weight_shape

    def chain_node(self, node, weights):
        """
        :param node:
            shape == (バッチサイズ, マップ数, 行数, 列数) なノード。
        """
        out_shape, weight_shape = self.calc_weight_shape(node)
        bias_shape = weight_shape[0]

        self.build_param(weight=weight_shape, bias=bias_shape)
        self.weight = self.params['weight']
        self.bias = self.params['bias']

        unit = conv2d(
            input=node.symbol,
            filters=self.weight.symbol,
            filter_shape=self.weight.value.shape,
            image_shape=node.shape,
            border_mode=self._mode
            )
        bias = self.bias.symbol.dimshuffle('x', 0, 'x', 'x')
        return Node(out_shape, self.act.act(unit + bias))

    def init_param(self, rng):
        """
        パラメータを乱数生成器で初期化。
        _Base.chain() の後で呼ばないと、 self.weight が見つからない。

        :param rng:
            乱数生成器。 numpy.random.RandomState() のインスタンス。
        """
        bound = numpy.sqrt(6. / numpy.prod(self.weight.value.shape))
        self.weight.init_value(rng, bound)

    def detail_str(self):
        """
        詳細情報としてフィルタ数, 活性化関数, mode("valid"か"full")を表示する。
        """
        return 'filter:{0} act:{1}'.format(self.filter_shape, self.act)


class Deconvolution(Convolution):
    """
    conv2d(mode='full') 以外は全部 Convolution と同じ

    Convolution の auto encoder を構成したいときに使う。
    Convolution が フィルタサイズが 5 のときに、
    出力マップのサイズが 5-1 だけ減るのに対して、
    Deconvolution では出力マップのサイズが 5-1 だけ増える。

    これを使って次元を合わせることができる。
    ::

        node = Node((10, 3, 23, 28))

        layers = (
            layer.DropOut(0.2),
            layer.Convolution((16, 4, 7), layer.Sigmoid(), tune=True),
            layer.Deconvolution((3, 4, 7), layer.Sigmoid(), tune=True),
        )

    このようにすると、バッチサイズ 10 で、23行28列の3チャンネル画像
    (カラー画像など) に、16個の 4行7列のフィルタを畳み込む Convolution 層の、
    Denoising Auto Encoder を構成できる。
    """
    _mode='full'


class SeparableConvolution(Convolution):
    """
    Convolutional 層の重み行列をそれぞれ rank1 に制限して学習し、
    推定を高速化する。
    """
    def chain_node(self, node, weights):
        """
        :param node:
            shape == (バッチサイズ, マップ数, 行数, 列数) なノード。
        """
        out_shape, weight_shape = self.calc_weight_shape(node)
        bias_shape = weight_shape[0]

        horizontal_shape = list(weight_shape)
        horizontal_shape[2] = 1
        vertical_shape = list(weight_shape)
        vertical_shape[3] = 1

        self.build_param(
            horizontal=horizontal_shape,
            vertical=vertical_shape,
            bias=bias_shape
        )
        self.horizontal = self.params['horizontal']
        self.vertical = self.params['vertical']
        self.bias = self.params['bias']
        horizontal = self.horizontal.symbol.repeat(weight_shape[2], axis=2)
        vertical = self.vertical.symbol.repeat(weight_shape[3], axis=3)
        filters = horizontal * vertical

        unit = conv2d(
            input=node.symbol,
            filters=filters,
            filter_shape=weight_shape,
            image_shape=node.shape,
            border_mode=self._mode
            )
        bias = self.bias.symbol.dimshuffle('x', 0, 'x', 'x')
        return Node(out_shape, self.act.act(unit + bias))

    def init_param(self, rng):
        """
        パラメータを乱数生成器で初期化。
        _Base.chain() の後で呼ばないと、 self.weight が見つからない。

        :param rng:
            乱数生成器。 numpy.random.RandomState() のインスタンス。
        """
        shapes = self.horizontal.shape, self.vertical.shape
        bound = numpy.sqrt(6. / numpy.prod(shapes))
        self.horizontal.init_value(rng, bound)
        self.vertical.init_value(rng, bound)

    def detail_str(self):
        """
        詳細情報としてフィルタ数, 活性化関数を表示する。
        """
        fmt = 'filter:{}({}*{}) act:{}'
        return fmt.format(
            self.filter_shape,
            self.horizontal.shape,
            self.vertical.shape,
            self.act
        )


class ElementWiseWeight(_Parametric):
    """
    各画素ごとに重み
    """
    def chain_node(self, node, weights):
        self.build_param(weight=node.shape[1:], bias=node.shape[1:])
        self.weight = self.params['weight']
        self.bias = self.params['bias']
        if 2 == node.ndim:
            weight = self.weight.symbol.dimshuffle('x', 0)
            bias = self.bias.symbol.dimshuffle('x', 0)
        elif 3 == node.ndim:
            weight = self.weight.symbol.dimshuffle('x', 0, 1)
            bias = self.bias.symbol.dimshuffle('x', 0, 1)
        else:
            assert 4 == node.ndim
            weight = self.weight.symbol.dimshuffle('x', 0, 1, 2)
            bias = self.bias.symbol.dimshuffle('x', 0, 1, 2)
        return Node(node.shape, node.symbol * weight + bias)

    def init_param(self, rng):
        self.weight.init_value(rng, 0.1, 1)


class LocallyConnected(Convolution):
    """
    cuda-convnet に実装されていて、 caffe に実装されていない、
    weights を共有しない畳み込み。

    これを、超変態的方法で実装。たぶん、速度は出ない。
    """
    def calc_weight_shape(self, node):
        assert 4 == len(node.shape), node
        batch_size, imaps, irows, icols = node.shape
        filters, frows, fcols = self.filter_shape
        inshp = irows, icols
        kshp = frows, fcols
        orows, ocols = ConvOp.getOutputShape(inshp, kshp, mode=self._mode)
        assert 0 < orows, orows
        assert 0 < ocols, ocols
        self.each_outs = filters, orows, ocols
        out_shape = batch_size, filters, orows, ocols
        weight_shape = filters, imaps, frows, fcols, orows, ocols
        return out_shape, weight_shape

    def chain_node(self, node, weights):
        """
        :param node:
            shape == (バッチサイズ, マップ数, 行数, 列数) なノード。
        """
        out_shape, weight_shape = self.calc_weight_shape(node)
        bias_shape = weight_shape[0]

        self.build_param(weight=weight_shape, bias=bias_shape)
        self.weight = self.params['weight']
        self.bias = self.params['bias']
        filter_shape = weight_shape[:4]
        in_shape = list(node.shape)
        in_shape[2] = self.filter_shape[1]
        in_shape[3] = self.filter_shape[2]

        symbols2d = []
        for r in xrange(out_shape[-2]):
            r_ = r + self.filter_shape[-2]
            symbols = []
            for c in xrange(out_shape[-1]):
                c_ = c + self.filter_shape[-1]
                in_symbol = node.symbol[:,:,r:r_,c:c_]
                symbol = conv2d(
                    input=in_symbol,
                    filters=self.weight.symbol[:,:,:,:,r,c],
                    filter_shape=filter_shape,
                    image_shape=in_shape,
                    border_mode=self._mode
                    )
                symbols.append(symbol)
            symbols2d.append(tensor.concatenate(symbols, axis=3))
        symbol = tensor.concatenate(symbols2d, axis=2)
        bias = self.bias.symbol.dimshuffle('x', 0, 'x', 'x')
        return Node(out_shape, self.act.act(symbol + bias))
