import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi

import wenet.dataset.kaldi_io as kaldi_io
import numpy as np
import json
import os

def parse_opts():
	parser = argparse.ArgumentParser(description='read_fbank')
	parser.add_argument('pos_dir', default=None,help='msg')
	parser.add_argument('neg_dir', default=None, help='msg')
	parser.add_argument('bla_dir', default=None, help='msg')
	args = parser.parse_args()
	return args

def init_data_list():
	"""
	return data: [(mat, label),...,()]

	mat 是一个 2D 数组，元素类型 float32，shape = (T ,80)， T 为时间帧长度，变化值。
	label 从 {0, 1} 取值
	"""
	# args = parse_opts()
	path = "G:\wenet_location\wav2vec\data"
	# print(os.path.exists(path))
	pos_dir = os.path.join(path ,"pos")
	neg_dir = os.path.join(path ,"neg")
	bla_dir = os.path.join(path ,"bla")

	dict_all = {}
	for dir in (pos_dir, neg_dir, bla_dir):
		file = os.path.join(dir, "wav.ark")
		# print("file is exists? {}".format(os.path.isfile(file)))
		# outFilt = dir + '/wav_fbank.json'
		d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
		dict_all.update(d)
	# print(dict_all['bs_0001'].shape, dict_all['na_01'].shape, dict_all['other_1'].shape)
	# print(len(dict_all))

	data = []
	for k, v in dict_all.items():
		label = 0
		if k.startswith('bs'):
			label = 1
		data.append((v, label))
		# print(data)
	return data

if __name__ == '__main__':
	print(len(init_data_list()))


