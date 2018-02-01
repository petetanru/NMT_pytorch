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
    src_name = 'src' if source else 'tgt'
    word_dict = {}
    master_set = set()
    for sample in file:
        set_letter = set(sample)
        master_set = master_set.union(set_letter)
    tokens = "ZERO UNK SOS EOS".split()
    master_set = master_set.difference(tokens)
    if vocab_type == 'bpe':
        tokens.append(' ')
    for ii, aa in enumerate(tokens):
        word_dict[aa] = ii
    print("initialized word %s dict with" % (src_name), word_dict)
    for ii, ww in enumerate(master_set):
        word_dict[ww] = ii + len(tokens)
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

''' preprocessing '''

def filter_data_len(data, target, mode):
    len_order = []
    x_len_order, y_len_order = [], []
    print(len(data), len(target))
    for i in range(len(target)):
        len_data = len(data[i])
        len_target = len(target[i])
        # if mode in ('c2c'):
        #     max_len = max((len_data, len_target))
        #     len_order.append(max_len)
        # elif mode in ('c2w','tcc2w', 'bpe2w', 'wp2w', 'w2w','bpe2bpe', 'w2bpe', 'c2bpe'):
        x_len_order.append(len_data)
        y_len_order.append(len_target)

    if mode in ('w2w', 'bpe2bpe', 'bpe2w', 'tcc2bpe'):
        x_three_list = zip(x_len_order, y_len_order, data, target)
        y_three_list = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in x_three_list if lx < 51 and lx > 0 and ly > 0]
        train_target = [y for lx, ly, x, y in y_three_list if lx < 51 and lx > 0 and ly >0]
    elif mode in ('c2w', 'c2bpe'):
        four_listx = zip(x_len_order, y_len_order, data, target)
        four_listy = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in four_listx if lx < 251 and ly < 51 and ly > 0]
        train_target = [y for lx, ly, x, y in four_listy if lx < 251 and ly < 51 and ly > 0]
    elif mode == 'c2c':
        x_three_list = zip(x_len_order, data, target)
        y_three_list = zip(x_len_order, data, target)
        train_data = [x for l, x, y in x_three_list if l < 251 and l > 0]
        train_target = [y for l, x, y in y_three_list if l < 251 and l > 0]
    elif mode in ('tcc2w', 'wp2w', 'w2bpe'):
        four_listx = zip(x_len_order, y_len_order, data, target)
        four_listy = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in four_listx if lx < 51 and ly < 51 and ly > 0]
        train_target = [y for lx, ly, x, y in four_listy if lx < 51 and ly < 51 and ly > 0]
    elif mode in ('bpe2c'):
        four_listx = zip(x_len_order, y_len_order, data, target)
        four_listy = zip(x_len_order, y_len_order, data, target)
        train_data = [x for lx, ly, x, y in four_listx if lx < 51 and ly < 251 and ly > 0]
        train_target = [y for lx, ly, x, y in four_listy if lx < 51 and ly < 251 and ly > 0]
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

def sent2vec(x, y, inp_dict, tgt_dict):
    vec_x, vec_y = [], []
    for idx, sample in enumerate(x):
        sent_array = []
        sent_array.append(inp_dict['SOS'])
        for vocab_x in sample:
            try:
                vec_word = inp_dict[vocab_x]
            except:
                vec_word = inp_dict['UNK']
            sent_array.append(vec_word)
        sent_array.append(inp_dict['EOS'])
        vec_x.append(sent_array)

    for idx, sample in enumerate(y):
        sent_array = []
        for vocab_y in sample:
            try:
                vec_word = tgt_dict[vocab_y]
            except:
                vec_word = tgt_dict['UNK']
            sent_array.append(vec_word)
        sent_array.append(tgt_dict['EOS'])
        vec_y.append(sent_array)

    return vec_x, vec_y

def vec2tensor(x_vec, y_vec):
    lengths_x = [len(s) for s in x_vec]
    lengths_y = [len(s) for s in y_vec]
    N = len(x_vec)
    idx = [i for i in range(N)]

    max_len_x = max(lengths_x)
    max_len_y = max(lengths_y)

    # zip, sort by len, unzip
    zip_vec = sorted(zip(lengths_x, lengths_y, x_vec, y_vec, idx), reverse=True)
    lengths_x, lengths_y, vec_x, vec_y, sorted_idx = zip(*zip_vec)

    xtensor = torch.zeros(N, max_len_x).long()
    ytensor = torch.zeros(N, max_len_y).long()

    for idx, [s_x, s_y] in enumerate(zip(vec_x, vec_y)):
        xtensor[idx, :lengths_x[idx]] =  torch.LongTensor(s_x)
        ytensor[idx, :lengths_y[idx]] =  torch.LongTensor(s_y)
    return xtensor, ytensor, lengths_x, lengths_y, sorted_idx

def sent2vec_google(x, y, inp_dict, tgt_dict, lang):
    vec_x, vec_y = [], []
    for idx, sample in enumerate(x):
        sent_array = []
        sent_array.append(inp_dict['<%s>' % lang])
        for vocab_x in sample:
            try:
                vec_word = inp_dict[vocab_x]
            except:
                vec_word = inp_dict['UNK']
            sent_array.append(vec_word)
        sent_array.append(inp_dict['EOS'])
        vec_x.append(sent_array)

    for idx, sample in enumerate(y):
        sent_array = []
        for vocab_y in sample:
            try:
                vec_word = tgt_dict[vocab_y]
            except:
                vec_word = tgt_dict['UNK']
            sent_array.append(vec_word)
        sent_array.append(tgt_dict['EOS'])
        vec_y.append(sent_array)

    return vec_x, vec_y



''' Legacy '''

# def train_vectorize(x, y, inp_dict, tgt_dict, mode):
#     if mode in ('w2w', 'c2c'):
#         seq_len_x = seq_len_finder(x, y)
#         seq_len_y = seq_len_x
#     elif mode in('c2w','tcc2w', 'bpe2w', 'wp2w'):
#         seq_len_x = seq_len_finder(x)
#         seq_len_y = seq_len_finder(y)
#
#     Xtensor = torch.zeros(len(x), seq_len_x+1).long()
#     ytensor = torch.zeros(len(x), seq_len_y+1).long()
#
#     for i, seq in enumerate(x):
#         for t, char in enumerate(seq):
#             try:
#                 Xtensor[i, t] = inp_dict[seq[t]]
#             except:
#                 Xtensor[i, t] = inp_dict['UNK']
#         Xtensor[i, len(seq)] = inp_dict['EOS']
#     for i_y, seq_y in enumerate(y):
#         for t_y, char_y in enumerate(seq_y):
#             try:
#                 ytensor[i_y, t_y] = tgt_dict[seq_y[t_y]]
#             except:
#                 ytensor[i_y, t_y] = tgt_dict['UNK']
#         ytensor[i_y, len(seq_y)] = tgt_dict['EOS']
#     return Xtensor, ytensor, seq_len_x, seq_len_y
#
# def google_train_vectorize(x, y, inp_dict, tgt_dict, lang, mode):
#     if mode in ('w2w', 'c2c'):
#         seq_len_x = seq_len_finder(x, y)
#         seq_len_y = seq_len_x
#     elif mode in('c2w','tcc2w', 'bpe2w', 'wp2w'):
#         seq_len_x = seq_len_finder(x)
#         seq_len_y = seq_len_finder(y)
#     Xtensor = torch.zeros(len(x), seq_len_x+2).long()
#     ytensor = torch.zeros(len(x), seq_len_y+2).long()
#     for i, seq in enumerate(x):
#         for t, char in enumerate(seq):
#             Xtensor[i, 0] = inp_dict['<%s>' % lang]
#             try:
#                 Xtensor[i, t+1] = inp_dict[seq[t]]
#             except:
#                 Xtensor[i, t] = inp_dict['UNK']
#         Xtensor[i, len(seq)] = inp_dict['EOS']
#     for i_y, seq_y in enumerate(y):
#         for t_y, char_y in enumerate(seq_y):
#             try:
#                 ytensor[i_y, t_y] = tgt_dict[seq_y[t_y]]
#             except:
#                 ytensor[i_y, t_y] = tgt_dict['UNK']
#         ytensor[i_y, len(seq_y)] = tgt_dict['EOS']
#     return Xtensor, ytensor, seq_len_x, seq_len_y
#
# def test_vectorize(x, inp_dict, mode):
#     if mode in ('w2w', 'c2c'):
#         seq_len_x = seq_len_finder(x)
#     elif mode in('c2w','tcc2w', 'bpe2w', 'wp2w'):
#         seq_len_x = seq_len_finder(x)
#     Xtensor = torch.zeros(len(x), seq_len_x+1).long()
#     for i, seq in enumerate(x):
#         for t, char in enumerate(seq):
#             try:
#                 Xtensor[i, t] = inp_dict[seq[t]]
#             except:
#                 Xtensor[i, t] = inp_dict['UNK']
#         Xtensor[i, len(seq)] = inp_dict['EOS']
#     return Xtensor, seq_len_x

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

def mask_matrix(len_list):
    maxlen = max(len_list)
    bsz = len(len_list)
    A = torch.arange(0, maxlen).repeat(bsz, 1) #(N,maxW) - N rows of (0,1,..., maxlen)
    B = torch.Tensor(len_list).unsqueeze(1) # N,1) - to be broadcasted as (N, maxW) (3,3,..., 3)
    mask = A.lt(B) # compare each element of broadcasted matrix a < b, 1 if true.
    mask = Variable(mask.float().cuda())
    return mask

def mask_cnn(lengths, pool_stride):
    maxlen = max(lengths)
    bsz = len(lengths)
    excess_len = maxlen - torch.Tensor(lengths)
    zero_pool = np.floor(excess_len / pool_stride)
    mask_len = [i if i>-1 else 0 for i in zero_pool]

    new_seq_len = np.floor(maxlen/pool_stride)
    A = torch.arange(0, new_seq_len).repeat(bsz, 1) #(N,maxW) - N rows of (0,1,..., maxlen)
    B = [new_seq_len]*bsz
    B = torch.Tensor(B) - torch.Tensor(mask_len)
    B = B.unsqueeze(1)
    mask = A.lt(B)
    mask = Variable(mask.float().cuda())
    return mask

def decoder_mask(inputs):
    A = torch.zeros(inputs.size()).cuda()
    mask = A.lt(inputs.data.float())
    mask = Variable(mask.float().cuda())
    return mask


def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)