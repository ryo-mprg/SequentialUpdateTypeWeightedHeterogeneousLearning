#coding: UTF-8
"""
calc
===================================
サブタスクに与える重みの計算，更新を行うクラス
"""

import numpy
import logging


class calc(object):
	def __init__(self):
		"""
		初期化

		:task_lists: 学習するタスクのリスト
		:task_dicts: タスク名をstrキーとする重み(学習と同時に更新する)
		"""
		self.task_lists = ['facial', 'gender', 'age', 'race', 'smile']
		self.task_dicts = {'facial':1.0, 'gender':1.0, 'age':1.0, 'race':1.0, 'smile':1.0}
		

	def __getitem__(self, key):
		"""
		各タスクの重みを取得する
		"""
		return self.task_dicts[key]


	def calc_stability(self, task_error):
		"""
		学習誤差から安定度を算出する

		:task_error: タスクの学習誤差
		"""
		error_data = numpy.array(task_error)
		ave = numpy.average(error_data) 		
		stde = numpy.std(error_data)
		return ave + 3*stde	


	def calc_weight(self, task_error):
		"""
		各タスクの誤差関数に付与する重みの算出，更新をする

		:task_error:
			各タスクの学習誤差のリスト = [[facial_error, ...], [gender_error, ...], ...]
		"""
		sigmas = {}

		for i, key in enumerate(self.task_lists):
			error = self.calc_stability(task_error[i])
			sigmas.update({key:error})

		self.save_text(sigmas, 'sigmas.txt')

		count = 0
		for key, value in sorted(sigmas.items(), key = lambda x:x[1]):
			"""
			サブタスクの重みを更新する
			"""
			if count == 0:
				value_main = value
				key_main = key
				count = 1
			else:
				weight = self.task_dicts[key_main] * (value_main / value)
				self.task_dicts[key] = weight

		self.save_text(self.task_dicts, 'update_weights.txt')


	def save_text(self, data, text):
		"""
		各タスクの基準値，重み値等をテキストに書き込む(追加書き込み)

		:data: 各タスク名をstrキーとする値(μ+3σ，重み 等)
		:text: 書き込みをするテキスト名
		"""
		fmt = 'facial: {0} gender: {1} age: {2} race: {3} smile: {4}\n'
		text_data = fmt.format(data['facial'], data['gender'], data['age'], data['race'], data['smile'])
		save_text_file = text
		with open(save_text_file, 'a') as f:
			f.write(text_data)
			f.close()
			logging.info(save_text_file)

