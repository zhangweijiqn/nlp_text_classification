import os
import pandas as pd
import jieba.posseg as jp
import sys
root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
sys.path.append(root_path)

import time
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split

import h5py
import pickle

stopwords = {line.strip(): 1 for line in open('data/stopwords.txt', 'r', encoding='utf-8').readlines()}

label_sep = "、"

def getFilelist(path):
    filelist = []
    files = os.listdir(path)
    for f in files:
        if (f[0] == '.') or os.path.isdir(path + '/' + f):
            pass
        else:
            filelist.append(f)
    return filelist

def read_file(input_file):
    print('opening file: ', input_file)
    try:
        df = pd.read_csv(input_file, header=0, low_memory=False, encoding='utf-8')  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    except Exception as e:
        print("===================================> read_file " + input_file + " exception: " + str(e))
        return None
    print('len_csv', len(df))
    return df

def extract_cn(line):
    if not isinstance(line, str):
        return ''
    cop = re.compile("[^\u4e00-\u9fa5^，。！：；？]")
    line = re.sub('\r|\n|\t', '', line)
    return cop.sub("", line)

def set_df_text(df, field_list, cn_clean):
    if df is None or not set(field_list).issubset(set(df.columns)):
        print("during split field_list exception: ", set(field_list), set(df.columns),
                set(field_list).issubset(set(df.columns)))
        return
    df.dropna(axis=0, how='any', subset=field_list, inplace=True)
    for index, f in enumerate(field_list):
        if cn_clean:
            df[f] = df[f].apply(lambda x: extract_cn(x))
        if index == 0:
            df['text'] = df[f]
        else:
            df['text'] = df['text'] + df[f]
    print("set_df_text len:", len(df))
    return df

def fenci_string(data):
    # 对文档进行分词处理，采用默认模式
    seg_list = get_text_words(data)
    # 将分词后的结果用空格隔开，保存至本地。比如"我来到北京清华大学"，分词结果写入为："我 来到 北京 清华大学"
    return ' '.join(seg_list)

def get_text_words(text):
    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    words_cut = jp.cut(text)
    # words_flags = [w.flag for w in words_cut]     #词性
    words = [w.word for w in words_cut if w.flag in flags and w.word not in stopwords]

    return words

def fenci_dir_pandas(path, save_path, field_list=['content'], field_label='label', cn_clean=True):
    fileList = getFilelist(path)
    for ff in fileList:
        print("Using jieba on " + ff)
        df = read_file(path + '/' + ff)
        df = set_df_text(df, field_list, cn_clean)
        df.dropna(axis=0, how='any', subset=[field_label], inplace=True)
        df['text'] = df['text'].apply(lambda x: fenci_string(x))
        print('start to write fenci result to ', save_path)
        if field_label != '':
            # df['labels'] = df[field_label].apply(lambda x: x.strip())
            df['labels'] = df[field_label]
            f = open(save_path + "/" + ff + "-seg.txt", "w+", encoding='utf-8')
            f.write('labels,contents\n')
            for index, row in df.iterrows():
                line = str(row['labels']) + ',' + row['text'] + '\n'
                f.write(line)
            f.close()
        else:
            f = open(save_path + "/" + ff + "-seg.txt", "w+", encoding='utf-8')
            f.write('contents\n')
            for index, row in df.iterrows():
                line = row['text'] + '\n'
                f.write(line)
            f.close()

def read_dir(input_dir, is_subdir=False):
    l = 0
    df1 = pd.DataFrame()
    file_list = os.listdir(input_dir)
    for file in file_list:  # 遍历文件夹
        if os.path.isdir(input_dir + "/" + file):
            if is_subdir:
                sub_file_list = os.listdir(input_dir + "/" + file)
                print("打开文件：", input_dir + "/" + file, "文件数目", len(sub_file_list))
                for fs in sub_file_list:
                    if not os.path.isdir(input_dir + "/" + file + "/" + fs):
                        print('打开文件：', input_dir + "/" + file + "/" + fs)
                        df2 = read_file(input_dir + "/" + file + "/" + fs)
                        if df2 is not None:
                            l = l + len(df2)
                            df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并
        else:
            df2 = read_file(input_dir + "/" + file)
            if df2 is not None:
                l = l + len(df2)
                df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并
    # df1.to_csv("data.csv")
    print(len(df1))
    return df1

def load_txt(file):
    with open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.read()
        print("Load data from file (%s) finished !" % file)
    return lines

def load_dir(input_dir, suffix=''):
    docs = []
    file_list = os.listdir(input_dir)
    if suffix != '':
        file_list = [f for f in file_list if f.endswith(suffix)]
    for file in file_list:  # 遍历文件夹
        if os.path.isdir(input_dir + "/" + file):
            sub_file_list = os.listdir(input_dir + "/" + file)
            print("打开文件：", input_dir + "/" + file, "文件数目", len(sub_file_list))
            for fs in sub_file_list:
                if not os.path.isdir(input_dir + "/" + file + "/" + fs):
                    print('打开文件：', input_dir + "/" + file + "/" + fs)
                    doc = load_txt(input_dir + "/" + file + "/" + fs)
                    if doc is not None:
                        docs.append(doc)
        else:
            doc = load_txt(input_dir + "/" + file)
            if doc is not None:
                docs.append(doc)
    # df1.to_csv("data.csv")
    print(len(docs))
    return docs

PAD_ID = 0
UNK_ID = 1


def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def extract_cn(line):
    if not isinstance(line, str):
        return ''
    cop = re.compile("[^\u4e00-\u9fa5^，。！：；？]")
    line = re.sub('\r|\n|\t', '', line)
    return cop.sub("", line)


def transform_multilabel_as_multihot(label_list, label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1
    return result


def get_sampled_data(df, is_sampled):
    mt_cols = ['label', 'feature']
    if is_sampled:
        df_raw = df[mt_cols].sample(frac=0.01, random_state=666).reset_index(drop=True)
    else:
        df_raw = df[mt_cols].sample(frac=1.0, random_state=666).reset_index(drop=True)
    return df_raw


def split_dataset(df_raw):
    # 数据格式为两列，一列文本一列标签
    # print(df_raw.head(10))
    # split data0
    print(df_raw)
    train_set, x = train_test_split(df_raw,
                                    # stratify=df_raw['label'],
                                    test_size=0.1,
                                    random_state=45)
    val_set, test_set = train_test_split(x,
                                         # stratify=x['label'],
                                         test_size=0.5,
                                         random_state=43)

    return train_set, val_set, test_set


def transLabel2indexHot(label_str, label2index):
    label_list = []
    if label_str is None:   # not isinstance(label_str, str) or
        return label_list
    label_list = [label2index.get(x.strip(), PAD_ID) for x in str(label_str).split(label_sep) if x.strip()]
    print("======>", label2index, label_str, label_list)
    return transform_multilabel_as_multihot(label_list, len(label2index))


def transLabel2index(label_str, label2index):
    label_list = []
    if label_str is None:   # not isinstance(label_str, str) or
        return label_list
    label_idx = label2index.get(str(label_str).strip(), PAD_ID)
    print("======>label_idx", label_idx)
    return label_idx


def transWord2index(line, word2index):
    if not isinstance(line, str) or line is None:
        return []
    word_list = [word2index.get(x.strip(), UNK_ID) for x in line.split(' ') if x.strip()]
    # for w in line.split(' '):
    #     if w in word2index:
    #         if isinstance(word2index[w], int):
    #             words.append(word2index[w])
    #     else:
    #         print("warning: %s not in word indx" % w)
    return word_list


def get_X_Y(df, word2index, label2index, multi_label=False):
    X = []
    Y = []
    df.dropna(axis=0, how='any', subset=['labels', 'contents'], inplace=True)
    for index, row in df.iterrows():
        x = transWord2index(row['contents'], word2index)
        X.append(x)
        if multi_label:
            y = transLabel2indexHot(row['labels'], label2index)
        else:
            y = np.array([int(row['labels'])]).astype('int32')
        Y.append(y)
        if index < 3:
            print(index, row['contents'], row['labels'])
            print(index, x, y)
        if index % 100000 == 0:
            print(index, row['contents'], row['labels'])
            print(index, x, y)
    return X, Y

def get_X(df, word2index):
    X = []
    df.dropna(axis=0, how='any', subset=['contents'], inplace=True)
    for index, row in df.iterrows():
        x = transWord2index(row['contents'], word2index)
        X.append(x)
        if index < 3:
            print(index, row['contents'])
            print(index, x)
        if index % 100000 == 0:
            print(index, row['contents'])
            print(index, x)
    return X


def swap_index(source_dict):
    target_dict = {}
    for key,value in source_dict.items():
        target_dict[value] = key
    return target_dict

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.
    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)
    Credits: From Keras `pad_sequences` function.
    """
    print("pad input clolum[0] length",len(sequences))
    print("pad input clolum[1] length", len(sequences[0]))

    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def pad_sequence(sequence, maxlen=None, dtype='int32', padding='post',
                 truncating='post', value=0.):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    elif truncating == 'post':
        trunc = sequence[:maxlen]
    else:
        raise ValueError("Truncating type '%s' not understood" % truncating)

    if padding == 'post':
        x[:len(trunc)] = trunc
    elif padding == 'pre':
        x[:-len(trunc):] = trunc
    else:
        raise ValueError("Padding type '%s' not understood" % padding)
    return x

def shuffle_and_split(X, Y, valid_rate=0.05, test_rate=0.05):
    # shuffle, split,
    # print("Y:", Y)
    xy = list(zip(X, Y))
    random.Random(10000).shuffle(xy)
    X, Y = zip(*xy)
    X = np.array(X)
    Y = np.array(Y)
    num_examples = len(X)
    print("num_examples:", len(X), ";X.shape:", X.shape, ";Y.shape:", Y.shape)

    num_valid = int(num_examples * valid_rate)
    num_train = int(num_examples - (num_valid + num_examples *test_rate))
    train_X, train_Y=X[0:num_train], Y[0:num_train]
    vaild_X, valid_Y=X[num_train:num_train+num_valid], Y[num_train:num_train+num_valid]
    test_X, test_Y=X[num_train+num_valid:], Y[num_train+num_valid:]
    return train_X, train_Y, vaild_X, valid_Y, test_X, test_Y

def set_df_text(df, field_list, cn_clean):
    if df is None or not set(field_list).issubset(set(df.columns)):
        print("during split field_list exception: ", set(field_list), set(df.columns),
                set(field_list).issubset(set(df.columns)))
        return
    df.dropna(axis=0, how='any', subset=field_list, inplace=True)
    for index, f in enumerate(field_list):
        if cn_clean:
            df[f] = df[f].apply(lambda x: extract_cn(x))
        if index == 0:
            df['text'] = df[f]
        else:
            df['text'] = df['text'] + df[f]
    print("set_df_text len:", len(df))
    return df

def save_data(cache_file_h5py, cache_file_pickle, word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X,
              test_Y):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index, label2index), target_file)
