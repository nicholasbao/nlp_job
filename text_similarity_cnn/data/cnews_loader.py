# coding: utf-8

import sys
from collections import Counter

import numpy as np
import keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

import random
random.seed(10)
def read_file(filename):
    """读取文件数据"""
    contentsl, contentsr, labels = [], [], []
    with open_file(filename) as f:
        reader = f.readlines()
        for index, line in enumerate(reader):
            try:
                split_line = line.strip().split("\t")
                contentsl.append(list(native_content(split_line[1])))
                contentsr.append(list(native_content(split_line[2])))
                labels.append(list(native_content(split_line[3])))
            except:
                pass
        #乱序
        print("process_file is ",contentsl[6],contentsr[6],labels[6])
        contentsl = np.array(contentsl)
        contentsr = np.array(contentsr)
        labels = np.array(labels)
        index = [i for i in range(len(contentsl))]
        np.random.shuffle(index)
        contentsq1 = contentsl[index]
        contentsq2 = contentsr[index]
        labels12 = labels[index]
        print("process_file2 is ",contentsq1[6],contentsq2[6],labels12[6])
        print(labels12)
    return contentsq1, contentsq2, labels12


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id,max_length=20):
    """将文件转换为id表示"""
    contentsl, contentsr, labels = read_file(filename)

    data_idl, data_idr, label_id = [], [], []
    print("the length of read-fileis ",len(contentsl),len(contentsr),len(labels))
    for i in range(len(contentsl)):
        data_idl.append([word_to_id[x] for x in contentsl[i] if x in word_to_id])
        data_idr.append([word_to_id[x] for x in contentsr[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    xl_pad = kr.preprocessing.sequence.pad_sequences(data_idl, max_length)
    xr_pad = kr.preprocessing.sequence.pad_sequences(data_idr, max_length)
    y_pad = kr.utils.to_categorical(labels, num_classes=2)  # 将标签转换为one-hot表示
    return xl_pad, xr_pad, y_pad


def batch_iter(xl,xr, y, batch_size=64):
    """生成批次数据"""
    data_len = len(xl)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    xl_shuffle = xl[indices]
    xr_shuffle = xr[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield xl_shuffle[start_id:end_id], xr_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
