# 肠鸣音识别

简介：本文档整理了肠鸣音识别的过程，用于解释过程中步骤的功能以及具体的代码实现。

## 1 概述

肠鸣音是病人肠胃发出的具有一定周期性的轻微的蜂鸣，了解一段时间内待观察者体内肠鸣音的出现次数帮助医生更准确和快速的估计病人患有某些特性病的概率。最新论文表示肠鸣音的种类分为5种，区分特点是持续的时长、声音频率、幅度和波形。

肠鸣音识别任务定义为给定一段时长的音频，录制音频的环境是一名待观察病人暂住的安静的病房，音频文件记录了该病人身体发出的肠鸣音和病房环境下的微弱噪声，将该音频文件喂给预训练好的肠鸣音识别模型，模型输出文件内5种肠鸣音各自出现的次数，以及每个肠鸣音出现的时间下标点。

肠鸣音识别任务的需求是训练一个复杂的特性声音识别模型或系统。

为了更容易地上手和完成该任务，将原本的复杂系统的一部分功能拆分出来：检测肠鸣音。

检测肠鸣音定义为给定一个时常极短的音频文件，仅仅包含肠鸣音帧和其两端的空白时间帧，将该音频文件喂给预训练的模型，模型输出的结果是该音频文件是否包含肠鸣音。一个空白噪声音频文件喂给模型，期望模型输出该音频文件不包含肠鸣音。该任务是一个并不复杂的二分类任务，后续可扩展二分类为五分类，以便满足五种肠鸣音的识别需求。

下面介绍检测肠鸣音的思路过程以及代码实现。

## 2 数据准备

本小节介绍肠鸣音数据格式，展示若干样本示例。使用 wenet 提供的 api 计算音频文件的每帧的fbank特征，并保存为特定的ark二进制格式。同样使用 wenet 提供的 api 读取 ark 格式文件获取到每个音频文件对应的的 fbank 特征矩阵，矩阵的维度为 `T*80` ，其中 T 为音频文件的时间帧长，80 为指定的 fbank 特征个数，它是可变化的超参数。最后，参照 paddlepaddle 提供的自定义数据集封装，所有音频样本的特征矩阵以及对应的标签（标签指明是否是音频文件，是音频文件标签为1，否标签为0）封装为自定义数据集，为模型训练提供友好支持。

### 2.1 数据源格式

源数据集总共具有的音频文件样本数量为 2041 个，其中具有肠鸣音的样本数量为 1901 个，命名格式为 `bs_0001.wav` ... `bs_1952.wav` ，文件命名不连续。不具有肠鸣音的样本数量为 140 个，其中噪音样本数量为 130 个，命名格式为 `other_1.wav` ... `other_131.wav`，空白音频样本数量为 10 个，明明格式为 `na_01.wav` ,..., `na_10.wav`。

下面给出肠鸣音，噪声，空白音频示例。

**bs_0003.wav**

![image-20220402093826512](https://github.com/MomentOfTime/Bowel_sounds_recognition/blob/main/images/image-20220402093826512.png)

**other_3.wav**

![image-20220402093655786](https://github.com/MomentOfTime/Bowel_sounds_recognition/blob/main/images/image-20220402093655786.png)

**na_03.wav**

![image-20220402093901312](https://github.com/MomentOfTime/Bowel_sounds_recognition/blob/main/images/image-20220402093901312.png)

### 2.2 提取fbank特征

我们使用 wenet 提供的 api 来实现提供fbank特征，让我们来看看是如何实现的。

新建 wav2vec 文件夹作为主要的工作目录，wav2vec 目录下的结构如下：

![image-20220402094331521](https://github.com/MomentOfTime/Bowel_sounds_recognition/blob/main/images/image-20220402094331521.png)

上图是在 wav2vec 目录下，使用 linux tree 命令打印的目录结构，从图中可以看出wav2vec目录下有两个文件夹和三个文件：

- data 文件夹，存放数据集统计信息与结构信息。pos 为正样本，neg 为噪音样本，bla 为空白音频样本。
- local 文件夹，存放数据处理脚本，供 run.sh 调用
- path.sh 为环境准备脚本，在 run.sh 中的前几行代码调用，为后续提供环境支持，包含 wenet 环境。
- run.sh 为主要的工作脚本，脚本内按照 stage 划分为程序化的若干步骤。
- tools 为软链接，指向 wenet/tools/ ，可通过访问该目录下的tools，间接访问 wenet/tools。

**run.sh 脚本内容如下：**

```shell
#!/bin/bash

# 准备程序环境，将 wenet 环境准备好，
. ./path.sh || exit 1;

stage=0
stop_stage=5

# 指明 data/pos data/neg data/bla 为我们存放信息的目录
dir=data
pos_dir=$dir/pos
neg_dir=$dir/neg
bla_dir=$dir/bla

# 配置传参的功能，可以通过 --args_name args_value 的方法，将参数传递给下一个脚本
. tools/parse_options.sh || exit 1;

# stae -1 第 "-1" 步，调用 local/data_preparation.sh 功能是准备数据，解释见下文
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
	local/data_preparation.sh
fi

# stage 0 第 “0” 步，对三个目录，循环调用 tools/compute_fbank_feats.py 传入三个参数
# 第一个参数 wav.scp 是输入，音频的标识与路径表
# 第二个参数 wav.ark 是输出，存储 fbank 二进制格式文件的路径
# 第三个参数 ark.scp 是输出，存储每个音频文件对应的 fbank 索引
# 结果是三个目录下均生成 wav.ark 和 ark.scp
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	for dir in $pos_dir $neg_dir $bla_dir; do
		python tools/compute_fbank_feats.py  $dir/wav.scp $dir/wav.ark $dir/ark.scp
	done
fi

# stage 1 第 “1” 步，调用 local/read_fbank.py 传入三个目录作为参数，解释见下文
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	python local/read_fbank.py $pos_dir $neg_dir $bla_dir
fi

```

**local/data_preparation.sh：** 

```shell
#!/bin/bash

wav_path=/mnt/g/wenet_location/asr-data/bs_data/wav_data

dir=../data
positive_dir=$dir/pos
negative_dir=$dir/neg
blank_dir=$dir/bla
tmp_dir=$dir/tmp

# 创建目录
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

```

新建 data/pos，data/neg，data/bla 三个目录，并为三个目录分别生成 `wav.scp` 文件，该文件存储了一张表，表结构为 `<key>` `<path>` ，key 为音频文件的标识，path 为音频文件的路径。

**data/pos/wav.scp：**

![image-20220402100651014](https://github.com/MomentOfTime/Bowel_sounds_recognition/blob/main/images/image-20220402100651014.png)

解释脚本 read_fbank.py，脚本内容如下：

**read_fbank.py**

```python
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
    # 分别对三个目录下的wav.ark读文件，并将全部数据存入字典 dict_all，
	for dir in (pos_dir, neg_dir, bla_dir):
		file = os.path.join(dir, "wav.ark")
		# print("file is exists? {}".format(os.path.isfile(file)))
		# outFilt = dir + '/wav_fbank.json'
        # d 为单个目录下的存储 fbank 特征矩阵的字典，key 为音频标识，例如"bs_0001"，mat 为"T*80" 维度的特征矩阵
		d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
		# update 将 d 字典添加进入 dict_all 字典
        dict_all.update(d)
	# print(dict_all['bs_0001'].shape, dict_all['na_01'].shape, dict_all['other_1'].shape)
	# print(len(dict_all))
	
    # 遍历 dict_all 字典，为 key以 “bs” 开头的样本标记为1，否则标记为0，返回数据格式为二元组的列表，例如data 为 [(特征矩阵_1， label_1),...,(特征矩阵_N，label_N)]
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



```

init_data_list 为所需的主要方法，无需参数，会根据 stage 0 步生成的 wav.ark 文件，返回所有数据的特征矩阵及其对应的标签。

如上所示，音频文件的 fbank 特征已经计算好了，并且为每个音频标记了对应的label，最后的数据格式为二元组的列表，我们还需要将其封装为适合 paddlepaddle 训练模型的数据集格式。

### 2.3 数据集封装

创建 jupyter python3 文件，local/model.ipynb，文件内容如下。**该小节，请配合 local/model.ipynb 或 model.pdf 阅读。**

```
import sys
sys.path.append('G:/wenet_location/wenet/')
import paddle
from read_fbank import init_data_list
from paddle.io import Dataset, DataLoader
import numpy as np
```

导入环境包，sys.path.append 将 wenet 目录加入系统环境变量，方便后面调用 wenet 下的 python 代码，原因是 read_fbank 中调用了 wenet 代码。

导入 local/read_fbank 提供的函数，init_data_list，即是上文中返回的二元组的列表的函数。

导入 paddle 提供的 Dataset 数据集和 DataLoader 数据读取迭代器。

```
 data_list = init_data_list()
```

拿到 二元组列表，命名为 data_list 。

```python
print(":{}".format(len(data_list)))
for index, data in enumerate(data_list):
    if index > 5 :
    	break
    print("idx={}, shape={}, label={}".format(index, data[0].shape, data[1]))
```

浏览 data_list 中内容，查看前五个数据样本的矩阵维度，标签。可以看到 shape=(T, 80)，T大小是变动的，因此，我们需要将 T 都进行0值填充到 T的最大值。

```python
print(min(data[0].shape[0] for data in data_list))
print(max(data[0].shape[0] for data in data_list))
```

查看 T 的取值范围为 [10, 223]

```python
NUM_SAMPLES=len(data_list)
BATCH_SIZE = 64
BATCH_NUM = NUM_SAMPLES // BATCH_SIZE
train_offset = int(NUM_SAMPLES * 0.6)
val_offset = int(NUM_SAMPLES * 0.8)
print(train_offset, val_offset)
```

定义一些数据集的配置属性值，

- NUM_SAMPLES 是数据集的样本总数
- BATCH_SIZE 是批大小为64，通常会将完整的数据集划分独立的多个批，在训练过程中，每次给模型喂入一个批，换句话说，在训练过程的一个周期epoch中，依次在每个批上，模型会计算预测值，损失值，并更新模型参数。
- BATCH_NUM 是数据集划分的批个数
- train_offset 是数据集中占前 0.6 比例的数据样本被视为训练数据
- val_offset 是数据集中占 0.6-0.8 比例的数据样本被视为验证数据
- 没有定义test_offset，但是数据集中占 0.8-1.0 比例的数据样本被视为测试数据

```python
class MyDataset(Dataset):
    """
    paddle.io.Dataset
    """
    def __init__(self, mode='train'):
        """

        """
        super(MyDataset, self).__init__()
        # 每次数据都会洗牌，保证训练，验证，测试数据集的 样本分布均衡
        np.random.shuffle(data_list)
        if mode == 'train':
            self.data_list = data_list[0: train_offset]
            pass
        elif mode == 'val':
            self.data_list = data_list[train_offset: val_offset]
            pass
        elif mode == 'test':
            self.data_list = data_list[val_offset:]
            pass
        else:
            print("mode should be in ['train', 'test', 'val']")
        self.num_samples = len(self.data_list)

    def __getitem__(self, index):
        """
        __getitem__index
        """
        # 样本 T 补全代码，
        data = self.data_list[index][0]
        # 计算 T 与 223 的差值
        padlen = 223 - data.shape[0]
        # 调用 np.pad 在 T 轴上，在后面补上 padlen 长度的 0 值
        data = np.pad(data, ((0,padlen),(0,0)))
        label = np.array(self.data_list[index][i], dtype=np.int64)
        return data, label
	
    def __len__(self):
        """
        __len__
        """
        return self.num_samples
  
train_dataset = MyDataset(mode='train')
test_dataset = MyDataset(mode='test')
val_dataset = MyDataset(mode='val')
print('=============train_dataset len is {} ============='.format(len(train_dataset)))
for data, label in train_dataset:
print(data.shape, label)
break
print('=============test_dataset len is {} ============='.format(len(test_dataset)))
for data, label in test_dataset:
print(data.shape, label)
break
print('=============val_dataset len is {} ============='.format(len(val_dataset)))
for data, label in val_dataset:
print(data.shap
```

上面代码定义的三个数据集，train_dataset，test_dataset，val_dataset，并且补全了每个数据样本的维度，每条样本形状shape 为 (223, 80)，223为帧最大值，80为fbank特征数量。可以看到，train_dataset有1124样本，test_dataset 有 409，val_dataset 有408。

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
```

数据集封装为数据读取迭代器，每次读取 64 个批大小的样本，shuffle = True 意味着读取数据时也会洗牌，drop_last = True 意味着？？？ 【待填坑】

数据准备工作完成，下面将介绍模型组网的内容。

## 3 模型组网

本小节将介绍模型的网络结构。

### 3.1 模型网络结构

代码内容如下：

```python
from paddle.nn import Layer, Linear, AdaptiveAvgPool1D, Softmax, CrossEntropyLoss
import paddle.nn.functional as F
class MyNet(Layer):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(80, 128)
        self.avgpool1 = AdaptiveAvgPool1D(output_size=1)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 2)

    def forward(self, inputs):
        # inputs.shape = (B, T, L) BTLfbank 此例中 (64, 223, 80)
        # y.shape = (B, L2) B L2
        # 用 0 初始化中间值 y， 此例中 形状为 （64, 223 , 128） 
        y = paddle.zeros((inputs.shape[0],inputs.shape[1],128),dtype=paddle.float32)
        # 对 223 个时间帧，每个时间帧的 80 维度向量喂入 80->128 的线性变换层， 因此输出形状是 （64， 223， 128）
        for idx in range(inputs.shape[1]):
        	y[:,idx,:] = self.linear1(inputs[:,idx,:])
        # (64, 223, 128) 经过转置 -> (64, 128, 223)
        y = paddle.transpose(y, [0, 2, 1])
        # (64, 128, 223) 经过平均池化，223 维度变为 1维度，输出为 (64, 128, 1)
        y = self.avgpool1(y)
        # (64, 128, 1) -> (64, 128) 去掉最后一个维度
        y = y[:, :, 0]
        # (64, 128) 经过第二个 128->64 的线性变换， (64, 64) 
        y = self.linear2(y)
        # 经过 激活函数，形状不变
        y = F.relu(y)
        # (64, 64) 经过第三个 64 -> 2 的线性变换，(64, 2)
        y = self.linear3(y)
        # (64, 2) 返回值 y 形状
        return y
```

导入 paddle 的 Layer，让我们网络 MyNet 继承 Layer。导入 Linear 线性层， AdaptiveAvgPool1D 适应平均1维度池化层，CrossEntropyLoss 交叉熵损失函数，导入 paddle.nn.functional 主要提供 relu 激活函数。

init 初始化函数中，定义了 80 维度到 128 维度的线形层，输出维度为1 的平均池化层，128 维度到 64 维度的线性层，64 维度到 2 维度的线性层。

forward 函数是模型训练过程，主要的执行函数。输入参数是 inputs ，inputs 即是我们定义的数据集读取迭代器的一个批，即是 64个数据样本，因此 inputs 形状 shape = (B, T, L) ，B 是64，代表64个样本，T 是时间帧，因为我们进行了 0 填充，因此 T 恒等于最大的时间帧223，L 是 fbank 的特征数量，80。

```python
paddle.summary(MyNet(),(1,10,80))
```

展示输入数据形状为（1，10，80）时，经过模型处理过程中，形状的变化过程。

### 3.2 评估指标

精准率计算公式为：
$$
acc = \frac{tp}{tp + fp}\tag1
$$
其中，tp 为预测为1的样本中，预测正确的样本个数，fp为预测为1的样本中，预测失败的样本个数。精准率衡量了预测为1的正确的概率。

```python
class PrecisionSoft(paddle.metric.Metric):
    """
    1. paddle.metric.Metric
    """
    def __init__(self, name='PrecisionSoft'):
        """
        2.
        """
        super(PrecisionSoft, self).__init__()
        self.tp = 0
        self.fp = 0
        self._name = name
    def name(self):
        """
        3. name
        """
        return self._name
    def update(self, preds, labels):
        """
        5. updatebatch
        - `compute``update`
        - `compute`compute`update`
        """
        sample_num = labels.shape[0]
        preds = paddle.to_tensor(preds, dtype=paddle.float32)
        # print("preds={}".format(preds))
        preds = paddle.argsort(preds, descending=True)
        preds = paddle.slice(
        	preds, axes=[len(preds.shape) - 1], starts=[0], ends=[1])
        for i in range(sample_num):
            pred = preds[i, 0].numpy()[0]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fp += 1

    def accumulate(self):
        """
        6. accumulatebatch
        `update``accumulate`
        `fit`
        """
        # update
        ap = self.tp + self.fp
        return float(self.tp) / ap if ap != 0 else .0
    
    def reset(self):
        self.tp = 0
        self.fp = 0

```

召回率评估指标公式为：
$$
recall = \frac{tp}{tp+fn}\tag2
$$
其中，tp 为标签值为1的样本中预测为1的样本数量，fn为标签值为1的样本中预测为0的样本数量，recall 衡量了具有肠鸣音的样本集合上，模型预测正确的概率。

```
class RecallSoft(paddle.metric.Metric):
    """
    1. paddle.metric.Metric
    """
    def __init__(self, name='RecallSoft'):
        """
        2.
        """
        super(RecallSoft, self).__init__()
        self.tp = 0
        self.fn = 0
        self._name = name
    def name(self):
        """
        3. name
        """
        return self._name
    def update(self, preds, labels):
        """
        5. updatebatch
        - `compute``update`
        - `compute`compute`update`
        """
        sample_num = labels.shape[0]
        preds = paddle.to_tensor(preds, dtype=paddle.float32)
        # print("preds={}".format(preds))
        preds = paddle.argsort(preds, descending=True)
        preds = paddle.slice(
        	preds, axes=[len(preds.shape) - 1], starts=[0], ends=[1])
        for i in range(sample_num):
            pred = preds[i, 0].numpy()[0]
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp += 1
                else:
                    self.fn += 1

    def accumulate(self):
        """
        6. accumulatebatch
        `update``accumulate`
        `fit`
        """
        # update
        recall = self.tp + self.fn
        return float(self.tp) / recall if recall != 0 else .0
    
    def reset(self):
        self.tp = 0
        self.fn = 0
```

F1评估指标公式为：
$$
F1=\frac{2*acc*recall}{acc+recall}\tag3
$$
其中，acc为公式1定义的精准率，recall为公式2定义的召回率，F1指数权衡了精准率和召回率，可以作为最终的评估模型性能的指标。

```
class F1Soft(paddle.metric.Metric):
    """
    1. paddle.metric.Metric
    """
    def __init__(self, name='F1Soft'):
        """
        2.
        """
        super(F1Soft, self).__init__()
        self.tp1 = 0
        self.fn = 0
        self.tp2 = 0
        self.fp = 0
        self._name = name
    def name(self):
        """
        3. name
        """
        return self._name
    def update(self, preds, labels):
        """
        5. updatebatch
        - `compute``update`
        - `compute`compute`update`
        """
        sample_num = labels.shape[0]
        preds = paddle.to_tensor(preds, dtype=paddle.float32)
        # print("preds={}".format(preds))
        preds = paddle.argsort(preds, descending=True)
        preds = paddle.slice(
        	preds, axes=[len(preds.shape) - 1], starts=[0], ends=[1])
        for i in range(sample_num):
            pred = preds[i, 0].numpy()[0]
            label = labels[i]
            if label == 1:
                if pred == label:
                    self.tp1 += 1
                else:
                    self.fn += 1
            if pred == 1:
            	if pred == label:
            		self.tp2 += 1
            	else:
            		self.fp += 1

    def accumulate(self):
        """
        6. accumulatebatch
        `update``accumulate`
        `fit`
        """
        # update
        ap = self.tp2 + self.fp
        recall = self.tp1 + self.fn
        
        ap = float(self.tp2) / ap if ap != 0 else .0
        recall = float(self.tp1) / recall if recall != 0 else .0
        return 2 * (ap * recall) / (ap + recall) if (ap + recall) != 0 else .0
    
    def reset(self):
        self.tp1 = 0
        self.fn = 0
        self.tp2 = 0
        self.fp
```



## 4 训练与测试

本小节介绍模型训练过程，并展示测试结果

### 4.1 训练

```python
from paddle import Model
from paddle.optimizer import Adam
from paddle.metric import Accuracy, Precision, Recall

model = Model(MyNet())

model.prepare(Adam(learning_rate=0.001, parameters= model.parameters()),
              CrossEntropyLoss(), 
              [Accuracy(), PrecisionSoft(), RecallSoft(), F1Soft()])
model.fit(train_loader, val_loader, epochs=10, verbose=2)
```

导入 paddle 的模型高层API Model，可以将我们定义的网络 MyNet() 封装为更方便使用的模型 model。

调用提供的 model.prepare()，配置模型的属性：

- Adam优化器，指定学习率为0.001，优化参数为模型的三个线形层，总计18754个参数。
- CrossEntropyLoss损失函数，交叉熵损失，预测结果与标签值一致时，loss为0，不一致时loss>0。
- Accuracy 准确率，预测结果与label的匹配程度，Precision精准率，Recall召回率，F1 评估指标

调用model.fit 开始训练，传入四个参数：

- train_loader，训练集读取迭代器
- val_loader，验证机读取迭代器
- epochs，总循环次数
- verbose，日志文本打印格式

### 4.2 测试

在测试集上，模型预测数据样本的结果展示

代码如下：

```
# model.predict返回的result，对于一个测试样本，其对应的每个类别的概率，
result = model.predict(test_loader)
# 查看预测结果的形状
print(len(result), len(result[0]), result[0][0].shape)
# 对 result 中，表示每个类别概率的轴上，降序排序，
result = paddle.argsort(paddle.to_tensor(result), descending=True)
# 取出降序排序中，第一个概率最大的下标，0表示预测为不具有，1表示预测为有
result = paddle.slice(result, axes=[len(result.shape) - 1], starts=[0],ends=[1])
# 查看预测结果的形状
print(result.shape)
```

```
result = result[0,:,:,0]
# 查看预测结果
result
```

```
t_r = paddle.where(result == 0, paddle.ones(result.shape), paddle.zeros(result.shape))
# 查看有几个被预测为0， 避免模型全部预测为1。
print(paddle.sum(t_r)

```

可以看到，有12个测试样本被预测为0。