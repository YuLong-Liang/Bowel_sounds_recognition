#!/bin/bash

. ./path.sh || exit 1;

stage=0
stop_stage=5

dir=data
pos_dir=$dir/pos
neg_dir=$dir/neg
bla_dir=$dir/bla

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
	local/data_preparation.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	for dir in $pos_dir $neg_dir $bla_dir; do
		python tools/compute_fbank_feats.py  $dir/wav.scp $dir/wav.ark $dir/ark.scp
	done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	python local/read_fbank.py $pos_dir $neg_dir $bla_dir
fi

