import os
import pickle
import re
import rouge
import jieba
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
import collections.abc as container_abcs
int_classes = int
string_classes = str
from transformers import MT5ForConditionalGeneration, BertTokenizer


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            line = json.loads(l)
            summary, dialog = line['summary'], line['dialog']
            D.append((summary, dialog))
    return D


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_data(data, tokenizer, encoder_max_len=512, decoder_max_len=64, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for title, content in data:
        text_ids = tokenizer.encode(
            content, max_length=encoder_max_len, truncation='only_first')
        if flag and term == 'train':
            flag = False
            print(content)
        if term == 'train':
            summary_ids = tokenizer.encode(
                title, max_length=decoder_max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                        }
        elif term == "test":
            features = {'dialog': content,
                        'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                        }
        else:
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        'title': title
                        }

        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, tokenizer, Only_test=False):
    """准备batch数据
    """

    if Only_test:
        test_data = load_data(args.test_data_path)
        return get_dataloader(args, test_data, tokenizer, term="test")

    train_data = load_data(args.train_data_path)
    valid_data = load_data(args.valid_data_path)
    test_data = load_data(args.test_data_path)

    train_dataloader = get_dataloader(
        args, train_data, tokenizer, term="train", batch_size=args.batch_size)
    valid_dataloader = get_dataloader(
        args, valid_data, tokenizer, term="valid", batch_size=args.infference_batch_size)
    test_dataloader = get_dataloader(args, test_data, tokenizer, term="test", batch_size=args.infference_batch_size)

    return train_dataloader, valid_dataloader, test_dataloader


def get_dataloader(args, data, tokenizer, term, batch_size):

    cache_path = os.path.join(args.cache_path, term+".pkl")

    if args.cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            features = pickle.load(f)
            if term == "train" and args.train_data_nums > len(features):
                raise ValueError('当前预使用数据数量【大于】数据集当前数量。')
            features = features[:args.train_data_nums] if term == "train" else features
    else:
        features = create_data(
            data, tokenizer, args.encoder_max_len, args.decoder_max_len, term)
        if args.cache:
            if not os.path.exists(args.cache_path):
                os.mkdir(args.cache_path)
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)

        if term == "train" and args.train_data_nums > len(features):
            raise ValueError('当前预使用数据数量【大于】当前cache 数据数量。')
        features = features[:args.train_data_nums] if term == "train" else features

    DataSets = KeyDataset(features)
    shuffle = args.shuffle if term == "train" else False
    dataloader = DataLoader(DataSets,
                            batch_size=batch_size,
                            drop_last=True,
                            shuffle=shuffle,
                            collate_fn=default_collate)

    return dataloader
