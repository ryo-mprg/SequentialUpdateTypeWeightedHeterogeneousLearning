# -*- coding: UTF-8 -*-
"""
update
================================

学習率とか更新とかを扱うクラス
"""
__author__ = 'watasue'
__copyright__ = "Copyright (C) 2014 watasue. All Rights Reserved."
__license__ = 'MIT'

import logging

import numpy


class BaseUpdateCounter(object):
    """
    パラメータの更新回数を管理するクラス。
    学習率固定、途中のtgzなしという一番ベーシックな更新。
    """
    def __init__(self, max_update_count, learning_rate, resume_count = 0):
        """
        学習に関する設定値を初期化時に与える。

        :param max_update_count:
            パラメータの更新回数。 resume_count が1以上であれば、
            resume_count 回のパラメータ更新はすでに行われたものとする。
            したがって、学習時にパラメータ更新する回数は、
            max_update_count - resume_count 回。

        :param leraning_rate:
            学習率。 ∂E/∂W で勾配を求め、その勾配をどのくらいの割合で
            パラメータに反映させるかを指定する。

            経験的には、学習する問題の難しさによって、学習率を調整すべき。
            訓練時のエラーが下がる範囲で、なるべく大きな数値がよい。
            小さい数値だと、同じエラー率に到達するまでにパラメータ更新回数が
            増え、学習時間が長くなるから。

            たとえば、人体検出は、入力のバリエーションが大きいのに、
            出力は二値であり、これは非常に難しい問題。難しい問題の場合、
            ネットワーク(Network)の階層構造を深くしないと十分な精度が出ない。
            この場合、学習率は、0.001 などの小さい値がよい。

            階層が浅い場合は、学習率は、0.5, 0.1などの大きめの値でもよい。
            階層が深くても、出力が二値画像の場合は、出力次元が大きいため、
            二値出力ほど難しい問題ではない。そのため、階層構造が深くても、
            学習率は多少大きめでもいける。

        :param resume_count:
            学習を再開する場合に指定する、すでにパラメータ更新された回数。
            デフォルトは 0。
        """
        self.count = resume_count
        self.max_count = max_update_count
        self.learning_rate = learning_rate

    def update(self, *args):
        self.count += 1

    def is_over(self):
        """
        :return: 真ならば、学習終了
        """
        return self.max_count <= self.count

    def is_time_to_tgz(self):
        """
        :return: 途中で tgz 化しない
        """
        return False

    def get_learning_rate(self, *args, **kwargs):
        return self.learning_rate

    def __str__(self):
        fmt = '{0.__class__.__name__} update: {0.count}  learning_rate: {0.learning_rate:.3e}'
        return fmt.format(self)


class UpdateCounter(BaseUpdateCounter):
    """
    学習率固定、途中の tgz あり
    """
    def __init__(
        self,
        max_update_count,
        learning_rate,
        resume_count = 0,
        tgz_range = (),
        epoch = 0,
    ):
        """
        :param tgz_range:
            保存されているログとパラメータファイルを
            tar+gz で固めるタイミングの指定。

            3要素のタプル(start, stop, step)。
            各要素とも、単位はパラメータ更新回数。
            start から stop まで、 step 回ごとに tar+gz で固める。
            pythonのビルトイン関数 range と異なり、
            start, stop ともに tar+gz で固める回に含まれる。

            学習を進めると、パラメータファイルは、どんどん上書きされるので、
            過去のパラメータ・ログのスナップショットをとるには、tar+gz で
            固めるしかない。

            更新のたびにパラメータファイルの名前を変える、という方法も
            考えられるが、それだと生成されるファイル数が膨大になるので、
            必要なら tar+gz でスナップショットを固める、という保存の仕方を
            採用した。

            tgz_range が無効値の場合は、学習前と学習後の2回だけ
            tar+gz で固める。
            デフォルトは ()、つまり無効値。
        """
        BaseUpdateCounter.__init__(
            self,
            max_update_count,
            learning_rate,
            resume_count
        )
        self.learning_rate0 = learning_rate
        self.tgz_range = tgz_range
        self.epoch = epoch

    def __is_valid_tgz_range(self):
        """
        :return: tgz_range は有効値。
        """
        if 3 != len(self.tgz_range):
            return False
        start, stop, step = self.tgz_range
        if stop <= start:
            return False
        if stop - start < step:
            return False
        return True

    def is_time_to_tgz(self):
        """
        :return:
            真ならば、現在はパラメータファイルとログファイルを
            tar+gz でかためるタイミング
        """
        if not self.__is_valid_tgz_range():
            return False
        start, stop, step = self.tgz_range
        if self.count < start:
            return False
        if stop < self.count:
            return False
        return (self.count - start) % step == 0

    def is_epoch(self):
        """
        重みの更新を行うかどうか判断する
        return：真ならば重み値の更新を行う
        """
        if self.epoch == 0:
            return False

        rem = self.count % self.epoch

        if rem == 0:
            return True
        else:
            return False



class AnnealingUpdateCounter(UpdateCounter):
    """
    焼き鈍し(simulated anealing)法により学習率を減少
    """
    # 学習率の最小値
    eps = numpy.finfo(numpy.float32).eps

    def __init__(
        self,
        max_update_count,
        learning_rate,
        time_to_anneal,
        resume_count = 0,
        tgz_range = (),
    ):
        """
        :param time_to_aneal:
            焼き鈍し(simulated anealing)法により学習率を減少しはじめる時間。
            単位はパラメータ更新回数。

            simulated anealing とは、パラメータの更新回数に応じて徐々に
            学習率を減少させる方法。パラメータ更新を何度も繰り返して、
            訓練時エラーが下ってくると、これまでの学習率では、訓練時エラーの
            ばらつきが目立ってくるようになることが多い。このとき、適度に
            学習率を小さくしてやることで、ばらつきが抑えられ、かつより小さい
            訓練時エラーになりやすい。

            なお、この方法は、単に anealing と論文で書かれていたりする。
            ほかの表記方法もあったけど、忘れた。

            デフォルトは、 None。つまり、simulated anealingを行わない。
        """
        UpdateCounter.__init__(
            self,
            max_update_count,
            learning_rate,
            resume_count,
            tgz_range
        )
        assert 0 < time_to_anneal, time_to_anneal
        self.time_to_anneal = time_to_anneal
        self.__calc_learning_rate()

    def __calc_learning_rate(self):
        """
        学習率を計算する。start_annealing_count が設定されていれば、
        焼き鈍し(simulated annealing)による学習率の更新をする。
        ただし、焼き鈍しをしても、最小学習率を下回ることはない。
        """
        if self.count <= self.time_to_anneal:
            self.learning_rate = self.learning_rate0
        else:
            coeff = float(self.time_to_anneal) / self.count
            self.learning_rate = self.learning_rate0 * coeff
            self.learning_rate = max(self.eps, self.learning_rate)

    def update(self, *args):
        """
        パラメータの更新回数をインクリメントし、学習率を再計算する。
        パラメータが更新されるたびに呼び出すこと。
        """
        self.count += 1
        self.__calc_learning_rate()


class ThoroughlyUpdateCounter(UpdateCounter):
    """
    収束するまでずっと学習を続ける
    """
    INITIAL_MIN_ERROR = numpy.finfo(numpy.float32).max

    def __init__(
        self,
        learning_rate,
        tgz_step,
        resume_count = 0,
        expand_rate = 2.0,
        threshold_rate = 0.999,
        lowest_error = 4e-6,
        forgotten_rate = 1.0,
    ):
        """
        :param leraning_rate:
            学習率。 ∂E/∂W で勾配を求め、その勾配をどのくらいの割合で
            パラメータに反映させるかを指定する。

        :param tgz_step:
            パラメータとログを tar+gz で固めるタイミングの指定。
            また、最初のmilestone。

        :param resume_count:
            学習を再開する場合に指定する、すでにパラメータ更新された回数。
            デフォルトは 0。

        :param expand_rate:
            最小エラーがあったとき、パラメータ更新回数を現在の何倍伸ばすか
            ひょっとするとデフォルトの 2.0 倍は大きすぎるかもしれない。

        :param threshold_rate:
            現在の最小エラーに対する閾値の倍率。
            要するに最小エラーよりもちょっと小さい値が、
            次の最小エラーになるようにする。

        :param lowest_error:
            最小エラーの下限値。これより小さい値は無視。
            デフォルト値は (1/256)**2 くらい

        :param forgotten_rate:
            train_error の忘却率。train_error が瞬間的に低下するケースでは、
            ここに1未満の値を入れて、時間的に平滑化する(指数加重平滑化)。
        """
        assert 0 <= learning_rate
        assert 0 < tgz_step
        assert 0 <= resume_count
        assert 1.0 < expand_rate
        assert 0 < threshold_rate < 1
        assert 0 < forgotten_rate <= 1
        self.learning_rate = learning_rate
        self.tgz_step = tgz_step
        self.count = resume_count
        self.expand_rate = expand_rate
        self.threshold_rate = threshold_rate
        self.lowest_error = lowest_error
        self.forgotten_rate = forgotten_rate
        self.milestone = tgz_step
        self.min_error = self.INITIAL_MIN_ERROR
        self.has_min_error = False
        self.train_error = self.min_error

    def update(self, train_error):
        """
        パラメータの更新回数をインクリメントする。
        パラメータが更新されるたびに呼び出すこと。

        :param train_error:
            訓練時エラー。milestone までに最小エラーが更新されたら、
            milestone は expand_rate 倍に伸びる。
            lowest_error 以下の数値になったら、 milestone は更新しない。
        """
        self.train_error *= (1 - self.forgotten_rate)
        self.train_error += self.forgotten_rate * train_error
        self.train_error = max(self.lowest_error, self.train_error)
        if self.train_error < self.min_error:
            self.has_min_error = True
            self.min_error = self.train_error * self.threshold_rate
            self.min_error = max(self.lowest_error, self.min_error)
            fmt = 'min_error:{} at {}'
            logging.info(fmt.format(self.min_error, self.count))
        self.count += 1
        if self.count == self.milestone and self.has_min_error:
            # expand_rate 倍した回数以降の一番近い保存タイミング
            self.milestone *= self.expand_rate
            self.milestone += self.tgz_step - 1
            self.milestone = int(self.milestone)
            self.milestone = self.milestone / self.tgz_step * self.tgz_step
            self.has_min_error = False
            logging.info('milestone:{}'.format(self.milestone))

    def is_over(self):
        """
        :return: 真ならば、学習終了
        """
        return self.milestone <= self.count

    def is_time_to_tgz(self):
        """
        :return:
            真ならば、現在はパラメータファイルとログファイルを
            tar+gz でかためるタイミング
        """
        return self.count % self.tgz_step == 0


class SelectiveUpdateCounter(UpdateCounter):
    """
    複数の学習率から一定間隔(学習セット1ファイル分)ごとに一番良いのを選ぶ
    """
    def __init__(
        self,
        max_update_count,
        ten_to_the = 8,
        divide = 2,
        resume_count = 0,
        tgz_range = (),
        n_updates_try = 100,
    ):
        """
        :param ten_to_the:
            学習率の最小値を10のべき乗(ten to the XX)で与える。
            デフォルトは 8 つまり ten to the negative eighth(1e-8)。
            google で "10 to the negative eighth" と検索したら 1e-8 と出る。
        :param divide:
            10をいくつに分けるか。デフォルトは 2 つまり、学習率として
            1e-X と 3.16e-X の2パターンになる。
            等比級数的に分ける(式にすれば  10 ** (-i/devide))。
            もし 3 にすると、学習率として
            1e-X, 2.15e-X, 4.64e-X の3パターンになる。
        """
        UpdateCounter.__init__(
            self,
            max_update_count,
            0,
            resume_count,
            tgz_range
        )
        self.learning_rates = [
            10 ** (-float(i + 1)/divide)
            for i in xrange(divide * ten_to_the + 1)
        ]
        self.n_updates_try = n_updates_try

    def get_learning_rate(self, network, dataset):
        """
        学習率を選ぶ
        """
        for k in dataset:
            n_data = len(dataset[k].get_value(borrow=True))
        n_updates_try = min(n_data, self.n_updates_try)
        n_wins = 0
        max_n_wins = len(self.learning_rates) / 4
        min_error = numpy.inf
        for i, r in enumerate(self.learning_rates):
            network.push_param()
            function = network.get_trainer(r, dataset)[-1]
            # 複数回ループさせたほうが、1.0でない学習率も選択される。
            for i in xrange(n_updates_try):
                error = function(i)
            logging.info('rate:{}  error:{}'.format(r, error))
            if error < min_error:
                self.learning_rate = r
                min_error = error
                n_wins = 1
            else:
                n_wins += 1
            network.pop_param()
            if max_n_wins <= n_wins:
                break
        return self.learning_rate
