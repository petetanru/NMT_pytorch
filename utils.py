import string
import unicodedata
import re
import nltk
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pickle
import os
import numpy as np
from collections import OrderedDict
from tcc import tcc

''' loading data and dict '''

all_en_letters = string.ascii_lowercase + " \".,;:'-?!" + "\n"
th_pattern = re.compile(r"[^\u0E00-\u0E7F? ']|^'|'$|''")

def en_unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
        and c in all_en_letters
    )

def th_unicode(string):
    char_rem = re.findall(th_pattern, string)
    list_char_rem = [char for char in string if not char in char_rem]
    sent = ''.join(list_char_rem)
    sent = ' '.join(sent.split())
    return sent

def load_data_en_word(path):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            sent = en_unicodeToAscii(sent)
            output.append(nltk.word_tokenize(sent))
    return output

def load_data_en_char(path):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            output.append(en_unicodeToAscii(sent))
    return output

# # load only when building thai vocab, otherwise comment out to avoid init tensorflow
# import deepcut
# def load_data_th_word(path):
#     output = []
#     input_file = os.path.join(path)
#     with open(input_file, "r", encoding='utf-8') as f:
#         data = f.read().split('\n')
#         for sent in data:
#             sent = th_unicode(sent)
#             output.append(deepcut.tokenize(sent))
#     return output

def load_data_th_char(path):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            output.append(th_unicode(sent))
    return output

def load_data_th_tcc(path):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            raw_tcc = tcc(sent)
            reformat_tcc = raw_tcc.split('/')
            output.append(reformat_tcc)
    return output

def build_vocab_tcc(file1, src):
    # tweaked version of: https://github.com/nyu-dl/dl4mt-c2c/blob/master/preprocess/build_dictionary_char.py
    # TODO: figure out why the original function omit characters like Ã.
    word_dict = {}
    master_set = set()
    for sample in file1:
        set_letter = set(sample)
        master_set = master_set.union(set_letter)
    if src:
        # 0 -> ZERO
        # 1 -> UNK
        # 2 -> SOS
        # 3 -> EOS
        tokens = "ZERO UNK SOS EOS".split()
    else:
        tokens = "EOS UNK".split()

    for ii, aa in enumerate(tokens):
        word_dict[aa] = ii

    for ii, ww in enumerate(master_set):
        word_dict[ww] = ii + len(tokens)

    if src:
        with open('input.th-en.tcc.pkl', 'wb') as f:
            pickle.dump(word_dict, f)
    else:
        with open('tgt.th-en.tcc.pkl', 'wb') as f:
            pickle.dump(word_dict, f)
    return word_dict


def build_vocab_word(file1, src):
    # tweaked version of: https://github.com/nyu-dl/dl4mt-c2c/blob/master/preprocess/build_dictionary_char.py
    # TODO: figure out why the original function omit characters like Ã.
    word_dict = {}
    master_set = set()
    for sample in file1:
        set_letter = set(sample)
        master_set = master_set.union(set_letter)

    if src:
        # 0 -> ZERO
        # 1 -> UNK
        # 2 -> SOS
        # 3 -> EOS
        tokens = "ZERO UNK SOS EOS".split()
    else:
        tokens = "EOS UNK".split()

    for ii, aa in enumerate(tokens):
        word_dict[aa] = ii

    for ii, ww in enumerate(master_set):
        word_dict[ww] = ii + len(tokens)

    if src:
        with open('input.th-en.word.pkl', 'wb') as f:
            pickle.dump(word_dict, f)
    else:
        with open('tgt.th-en.word.pkl', 'wb') as f:
            pickle.dump(word_dict, f)
    return word_dict

def build_vocab_char(filename, src, lang):
    print('Processing', filename)
    word_freqs = OrderedDict()
    with open(filename, 'r', encoding='utf-8') as f:
        for number, line in enumerate(f):
            if number % 20000 == 0:
                print('line', number)
            if lang == 'EN':
                line = en_unicodeToAscii(line)
                # line = nltk.word_tokenize(line)
            elif lang == 'TH':
                line = th_unicode(line)

            words_in = list(line)
            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1

    print('count finished')

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    if src:
        # 0 -> ZERO
        # 1 -> UNK
        # 2 -> SOS
        # 3 -> EOS
        tokens = "ZERO UNK SOS EOS".split()
    else:
        tokens = "EOS UNK".split()

    for ii, aa in enumerate(tokens):
        worddict[aa] = ii
    print(worddict)

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + len(tokens)

    print('start dump')
    with open('%s.%d.pkl' % (filename, len(tokens)), 'wb') as f:
        pickle.dump(worddict, f)

    f.close()
    print('Done')
    print(len(worddict))
    return worddict

''' preprocessing '''

def filter_data_len(data, target, mode):
    len_order = []
    x_len_order, y_len_order = [], []
    for i in range(len(data)):
        len_data = len(data[i])
        len_target = len(target[i])
        if mode in ('w2w', 'c2c'):

            max_len = max((len_data, len_target))
            len_order.append(max_len)
        elif mode in ('c2w','tcc2w'):
            x_len_order.append(len_data)
            y_len_order.append(len_target)

#     three_list = sorted(zip(len_order, data, target))
    if mode == 'w2w':
        x_three_list = zip(len_order, data, target)
        y_three_list = zip(len_order, data, target)
        train_data = [x for l, x, y in x_three_list if l < 31 and l > 0]
        train_target = [y for l, x, y in y_three_list if l < 31 and l > 0]
    elif mode == 'c2w':
        four_listx = zip(x_len_order, y_len_order, data, target)
        four_listy = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in four_listx if lx < 251 and ly < 31 and ly > 0]
        train_target = [y for lx, ly, x, y in four_listy if lx < 251 and ly < 31 and ly > 0]
    elif mode == 'c2c':
        x_three_list = zip(len_order, data, target)
        y_three_list = zip(len_order, data, target)
        train_data = [x for l, x, y in x_three_list if l < 251 and l > 0]
        train_target = [y for l, x, y in y_three_list if l < 251 and l > 0]
    elif mode == 'tcc2w':
        four_listx = zip(x_len_order, y_len_order, data, target)
        four_listy = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in four_listx if lx < 51 and ly < 31 and ly > 0]
        train_target = [y for lx, ly, x, y in four_listy if lx < 51 and ly < 31 and ly > 0]
    else:
        raise Error
    return train_data, train_target

def seq_len_finder(*args):
    longest_sent = 0
    for arg in args:
        for sentence in arg:
            curr_len = len(sentence)
            if curr_len > longest_sent:
                longest_sent = curr_len
    return longest_sent

def train_vectorize(x, y, inp_dict, tgt_dict, mode):
    if mode in ('w2w', 'c2c'):
        seq_len_x = seq_len_finder(x, y)
        seq_len_y = seq_len_x
    elif mode in('c2w','tcc2w'):
        seq_len_x = seq_len_finder(x)
        seq_len_y = seq_len_finder(y)
    Xtensor = torch.zeros(len(x), seq_len_x+1).long()
    ytensor = torch.zeros(len(x), seq_len_y+1).long()
    for i, seq in enumerate(x):
        for t, char in enumerate(seq):
            try:
                Xtensor[i, t] = inp_dict[seq[t]]
            except:
                Xtensor[i, t] = inp_dict['UNK']
        Xtensor[i, len(seq)] = inp_dict['EOS']
    for i_y, seq_y in enumerate(y):
        for t_y, char_y in enumerate(seq_y):
            try:
                ytensor[i_y, t_y] = tgt_dict[seq_y[t_y]]
            except:
                ytensor[i_y, t_y] = tgt_dict['UNK']
        ytensor[i_y, len(seq_y)] = tgt_dict['EOS']
    return Xtensor, ytensor, seq_len_x, seq_len_y

''' pytorch utils '''

def pad1d(tensor, pad, permute_dims=True):
    '''
    source: https://github.com/pytorch/pytorch/issues/2637
    input tensor - shape (batch, time, feat)
    input pad - how many paddings for (left, right). can do asymmetrical
    '''
    if permute_dims:
        tensor = tensor.permute(0, 2, 1).contiguous() # get features on first dim since we are padding time
    else:
        tenosr = tensor.contiguous()
    original_size = tensor.size() # (batch, feat, time)
    final_new_size = (original_size[0], original_size[1], original_size[2] + pad[0] + pad[1])
    temp_new_size = original_size[:2] + (1,) + original_size[2:]
    assert len(temp_new_size) == 4
    tensor = tensor.view(*temp_new_size)
    pad = pad + (0, 0)
    tensor = F.pad(tensor, pad)
    tensor = tensor.view(*final_new_size)
    if permute_dims:
        tensor = tensor.permute(0, 2, 1)
    return tensor

def repackage_hidden(h):
    # Frees up variable from old graph. Variables
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)