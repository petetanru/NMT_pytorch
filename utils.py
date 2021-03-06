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
wp = {
    'th':'th-vi/bpe/th-vi_train_bpe_ready.th',
    'en':'',
    'vi':''
}

all_en_letters = string.hexdigits + string.ascii_lowercase + " \".,;:'-?!" + "\n"
th_pattern = re.compile(r"[^\u0E00-\u0E7F? 1-9']|^'|'$|''")

def vi_unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
    )

def th_unicode(string):
    char_rem = re.findall(th_pattern, string)
    list_char_rem = [char for char in string if not char in char_rem]
    sent = ''.join(list_char_rem)
    sent = ' '.join(sent.split())
    return sent

def en_unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
        and c in all_en_letters
    )

# load only when building thai vocab, otherwise comment out to avoid init tensorflow
def build_vocab(file, source, vocab_type, lang, lang_pair):
    # for word, tcc, and bpe
    # tweaked version of: https://github.com/nyu-dl/dl4mt-c2c/blob/master/preprocess/build_dictionary_char.py
    # TODO: figure out why the original function omit characters like Ã.
    word_dict = {}
    master_set = set()
    for sample in file:
        set_letter = set(sample)
        master_set = master_set.union(set_letter)
    if source:
        # 0 -> ZERO
        # 1 -> UNK
        # 2 -> SOS
        # 3 -> EOS
        tokens = "ZERO UNK SOS EOS".split()
    else:
        tokens = "EOS UNK".split()
    if vocab_type == 'bpe':
        tokens.append(' ')
    for ii, aa in enumerate(tokens):
        word_dict[aa] = ii
    for ii, ww in enumerate(master_set):
        word_dict[ww] = ii + len(tokens)
    src_name = 'src' if source else 'tgt'
    with open('%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, vocab_type, src_name, lang), 'wb') as f:
        pickle.dump(word_dict, f)
    return word_dict

def wp_encode_gen(data, lang):
    import collections
    import subword_text_tokenizer
    subword_tokenizer = subword_text_tokenizer.SubwordTextTokenizer()
    token_counts = collections.Counter()
    if lang == 'th':
        with open(wp['th'], 'r', encoding='utf-8') as f:
            bpe_file = f.read().split('\n')
            for bpe_sent in bpe_file:
                token_counts.update(bpe_sent.split())
        encoder = subword_tokenizer.build_to_target_size(8000, token_counts, 2, 10)
    else:
        raise NotImplementedError
    return encoder

def load_data_en(path, lang_pair, vocab_type, source, train):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            sent = en_unicodeToAscii(sent) # uncomment to accept all characters
            if vocab_type == 'w':
                output.append(nltk.word_tokenize(sent))
            elif vocab_type == 'c':
                output.append(sent)
    if train is True:
        en_dict = build_vocab(output, source, vocab_type, 'en', lang_pair)
        return output, en_dict
    return output

def load_data_th(path, lang_pair, vocab_type, source, train):
    output = []
    lang = 'th'
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        if vocab_type == 'w':
            import deepcut
            for sent in data:
                sent = th_unicode(sent)  # comment out to allow numbers
                output.append(deepcut.tokenize(sent))
        elif vocab_type == 'c':
            for sent in data:
                # sent = th_unicode(sent)
                output.append(sent)
        elif vocab_type == 'tcc':
            for sent in data:
                raw_tcc = tcc(sent)
                reformat_tcc = raw_tcc.split('/')
                output.append(reformat_tcc)
        elif vocab_type == 'bpe':
            for sent in data:
                # word_list = sent.replace('@@', '').split()
                word_list = sent.split()
                output.append(word_list)
        elif vocab_type == 'wp':
            encoder = wp_encode_gen(data, lang)
            for sent in data:
                new_sent = encoder.encode(sent)
                output.append(new_sent)
    if train is True:
        th_dict = build_vocab(output, source, vocab_type, lang=lang, lang_pair=lang_pair)
        return output, th_dict
    return output

def load_data_vi(path, lang_pair, vocab_type, source, train):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            if vocab_type == 'w':
                output.append(nltk.word_tokenize(sent))
            elif vocab_type == 'c':
                output.append(sent)
            elif vocab_type == 'bpe':
                word_list = sent.split()
                output.append(word_list)
    if train is True:
        vi_dict = build_vocab(output, source, vocab_type, 'vi', lang_pair)
        return output,vi_dict
    return output

''' preprocessing '''

def filter_data_len(data, target, mode):
    len_order = []
    x_len_order, y_len_order = [], []
    for i in range(len(target)):
        # print(len(data), len(target))
        len_data = len(data[i])
        len_target = len(target[i])
        if mode in ('w2w', 'c2c'):
            max_len = max((len_data, len_target))
            len_order.append(max_len)
        elif mode in ('c2w','tcc2w', 'bpe2w', 'wp2w'):
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
    elif mode in ('tcc2w', 'bpe2w', 'wp2w'):
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
    elif mode in('c2w','tcc2w', 'bpe2w', 'wp2w'):
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

def google_train_vectorize(x, y, inp_dict, tgt_dict, lang, mode):
    if mode in ('w2w', 'c2c'):
        seq_len_x = seq_len_finder(x, y)
        seq_len_y = seq_len_x
    elif mode in('c2w','tcc2w', 'bpe2w', 'wp2w'):
        seq_len_x = seq_len_finder(x)
        seq_len_y = seq_len_finder(y)
    Xtensor = torch.zeros(len(x), seq_len_x+2).long()
    ytensor = torch.zeros(len(x), seq_len_y+2).long()
    for i, seq in enumerate(x):
        for t, char in enumerate(seq):
            Xtensor[i, 0] = inp_dict['<%s>' % lang]
            try:
                Xtensor[i, t+1] = inp_dict[seq[t]]
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