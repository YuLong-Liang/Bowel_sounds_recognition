#!/bin/bash

wav_path=/mnt/g/wenet_location/asr-data/bs_data/wav_data


dir=../data
positive_dir=$dir/pos
negative_dir=$dir/neg
blank_dir=$dir/bla
tmp_dir=$dir/tmp

mkdir -p $dir
mkdir -p $tmp_dir
mkdir -p $positive_dir
mkdir -p $negative_dir
mkdir -p $blank_dir

find $wav_path -iname "*.wav" > $tmp_dir/wav.flist
# cat $tmp_dir/wav.flist

grep -i "positive" $tmp_dir/wav.flist > $dir/pos/wav.flist
grep -i "negative" $tmp_dir/wav.flist > $dir/neg/wav.flist
grep -i "blank" $tmp_dir/wav.flist > $dir/bla/wav.flist

for dir in $positive_dir $negative_dir $blank_dir; do
	sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
	paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
	sort -u $dir/wav.scp_all > $dir/wav.scp
done

echo "$0: data preparation succeeded"
