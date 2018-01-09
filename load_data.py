import pickle
import os
from utils import build_vocab, en_unicodeToAscii
import nltk

''' Path '''

th_en = {
    'train_raw_th' : 'th-en/ted_train_th-en.th',
    'train_raw_en' : 'th-en/ted_train_th-en.en',
    'test_raw_th' : 'th-en/ted_test_th-en.th',
    'test_raw_en' : 'th-en/ted_test_th-en.en',
    'train_loaded_th' : 'th-en/th-en_train_tokenized-loaded.th',
    'train_loaded_en' : 'th-en/th_en_train_tokenized-loaded.en',
    'test_loaded_th' : 'th-en/th-en_val_tokenized-loaded.th',
    'test_loaded_en' : 'th-en/th-en_val_tokenized-loaded.en',
    'train_bpe_th' : 'th-en/bpe/th-en_train.th.bpe60000.experiment',
    'test_bpe_th' : 'th-en/bpe/th-en_test.th.bpe60000.experiment',
    # 'train_bpe_th': 'th-en/bpe/th-en_train.th.bpe8192.experiment',
    # 'test_bpe_th': 'th-en/bpe/th-en_test.th.bpe8192.experiment'
    # 3 files for bpe: bpe8192, bpe16384, and bpe32768
}

th_en_sent = {
    'train_raw_th' : 'th-en_sent/ted_train_th-en.th',
    'train_raw_en' : 'th-en_sent/ted_train_th-en.en',
    'test_raw_th' : 'th-en_sent/ted_test_th-en.th',
    'test_raw_en' : 'th-en_sent/ted_test_th-en.en',
    'train_loaded_th' : 'th-en_sent/th-en_train_tokenized-loaded.th',
    'train_loaded_en' : 'th-en_sent/th-en_train_tokenized-loaded.en',
    'test_loaded_th' : 'th-en_sent/th-en_val_tokenized-loaded.th',
    'test_loaded_en' : 'th-en_sent/th-en_val_tokenized-loaded.en',
    'train_bpe_th' : 'th-en/th-en_train_bpe_ready.th.bpe',
    'test_bpe_th' : 'th-en_sent/th-en_test_bpe_ready.th.bpe',
    'train_bpe_en': 'th-en_sent/th-en_train_bpe_ready.en.bpe',
    'test_bpe_en': 'th-en_sent/th-en_test_bpe_ready.en.bpe'
    # 'train_bpe_th': 'th-en/bpe/th-en_train.th.bpe8192.experiment',
    # 'test_bpe_th': 'th-en/bpe/th-en_test.th.bpe8192.experiment'
    # 3 files for bpe: bpe8192, bpe16384, and bpe32768
}

th_vi = {
    'train_raw_th': 'th-vi/ted_train_th-vi.th',
    'train_raw_vi': 'th-vi/ted_train_th-vi.vi',
    'test_raw_th': 'th-vi/ted_test_th-vi.th',
    'test_raw_vi': 'th-vi/ted_test_th-vi.vi',
    'train_loaded_th': 'th-vi/th-vi_train_tokenized-nonum-seg.th',
    'train_loaded_vi': 'th-vi/th-vi_train_tokenized-nonum-seg.vi',
    'test_loaded_th': 'th-vi/th-vi_val_tokenized-nonum-seg.th',
    'test_loaded_vi': 'th-vi/th-vi_val_tokenized-nonum-seg.vi',
    # 'train_bpe_th': 'th-vi/bpe/th-vi_train.th.bpe8192.experiment',
    # 'test_bpe_th': 'th-vi/bpe/th-vi_test.th.bpe8192.experiment',
    'train_bpe_th': 'th-vi/bpe/th-vi_train.th.bpe60000.experiment',
    'test_bpe_th': 'th-vi/bpe/th-vi_test.th.bpe60000.experiment',
    'train_wp_th': 'th-vi/bpe/th-vi_train_bpe_ready.th',
    'test_wp_th': 'th-vi/bpe/th-vi_test_bpe_ready.th'
}

vi_en ={
    'train_raw_vi': 'vi-en/ted_train_vi-en.vi',
    'train_raw_en': 'vi-en/ted_train_vi-en.en',
    'test_raw_vi': 'vi-en/ted_test_vi-en.vi',
    'test_raw_en': 'vi-en/ted_test_vi-en.en',
    'train_loaded_vi': 'vi-en/ted_train_vi-en.vi.pkl',
    'train_loaded_en': 'vi-en/ted_train_vi-en.en.pkl',
    'test_loaded_vi': 'vi-en/ted_test_vi-en.vi.pkl',
    'test_loaded_en': 'vi-en/test_test_vi-en.en.pkl',
    'train_bpe_vi': 'vi-en/ted_train_vi-en.vi.bpe60000.experiment',
    'test_bpe_vi': 'vi-en/ted_test_vi-en.vi.bpe60000.experiment'
}

sent = True

''' functions '''
def load_data_th(path, lang_pair, vocab_type, source, train):
    output = []
    lang = 'th'
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        if vocab_type == 'w':
            import deepcut
            for sent in data:
                # sent = th_unicode(sent)  # comment out to allow numbers
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

def load_data_en(path, lang_pair, vocab_type, source, train):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().split('\n')
        for sent in data:
            # sent = en_unicodeToAscii(sent)
            if vocab_type == 'w':
                output.append(nltk.word_tokenize(sent))
            elif vocab_type == 'c':
                output.append(sent)
            elif vocab_type == 'bpe':
                word_list = sent.split()
                output.append(word_list)
    if train is True:
        en_dict = build_vocab(output, source, vocab_type, 'en', lang_pair)
        return output, en_dict
    return output

''' execution '''

def loader(train_loaded, val_loaded, dict_src):
    with open(train_loaded, 'rb') as f1:
        train_data = pickle.load(f1)
    with open(val_loaded, 'rb') as f2:
        val_data = pickle.load(f2)
    with open(dict_src, 'rb') as f3:
        inp_dict = pickle.load(f3)
    return train_data, val_data, inp_dict

def maker(train_raw, test_raw, lang_loader, lang_pair, vocab_type, source):
    train_data, dict = lang_loader(train_raw, lang_pair, vocab_type, source, train=True)
    val_data = lang_loader(test_raw, lang_pair, vocab_type, source, train=False)
    return train_data, val_data, dict

def saver(train_data, val_data, train_load, test_load):
    with open(train_load, 'wb') as f4:
        pickle.dump(train_data, f4)
    with open(test_load, 'wb') as f5:
        pickle.dump(val_data, f5)

''' code '''

def load_data(lang_pair, source_type, tgt_type):
    if lang_pair == 'th-en':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        source_loader = load_data_th
        if source_type == 'w' and sent is True:
            try:
                train_data, val_data, inp_dict = \
                    loader(th_en_sent['train_loaded_th'],
                       th_en_sent['test_loaded_th'],
                       source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(th_en_sent['train_raw_th'],
                          th_en_sent['test_raw_th'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_data, val_data,
                      th_en_sent['train_loaded_th'],
                      th_en_sent['test_loaded_th'])
        elif source_type =='w' and sent is False:
            try:
                train_data, val_data, inp_dict = \
                    loader(th_en['train_loaded_th'],
                           th_en['test_loaded_th'],
                           source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(th_en['train_raw_th'],
                          th_en['test_raw_th'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_data, val_data,
                      th_en['train_loaded_th'],
                      th_en['test_loaded_th'])
        elif source_type == 'c' and sent is True:
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_raw_th'],
                      th_en_sent['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'c' and sent is False:
            train_data, val_data, inp_dict = \
                maker(th_en['train_raw_th'],
                      th_en['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'tcc':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_raw_th'],
                      th_en_sent['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_bpe_th'],
                      th_en_sent['test_bpe_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'wp':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_bpe_th'],
                      th_en_sent['test_bpe_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        source_loader = load_data_en
        if tgt_type == 'w' and sent is True:
            try:
                train_target, val_target, tgt_dict = \
                    loader(th_en_sent['train_loaded_en'],
                       th_en_sent['test_loaded_en'],
                       tgt_dict_path)

            except:
                train_target, val_target, tgt_dict = \
                    maker(th_en_sent['train_raw_en'],
                          th_en_sent['test_raw_en'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_target, val_target,
                      th_en_sent['train_loaded_en'],
                      th_en_sent['test_loaded_en'])
        elif tgt_type == 'w' and sent is False:
            try:
                train_target, val_target, tgt_dict = \
                    loader(th_en['train_loaded_en'],
                       th_en['test_loaded_en'],
                       tgt_dict_path)

            except:
                train_target, val_target, tgt_dict = \
                    maker(th_en['train_raw_en'],
                          th_en['test_raw_en'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_target, val_target,
                      th_en['train_loaded_en'],
                      th_en['test_loaded_en'])
        elif tgt_type == 'c':
            train_target, tgt_dict = load_data_en(th_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                             source=source, train=True)
            val_target = load_data_en(th_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                            source=source, train=False)
        elif tgt_type == 'bpe' and sent is True:
            train_target, val_target, tgt_dict = \
                maker(th_en_sent['train_bpe_en'],
                      th_en_sent['test_bpe_en'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        else:
            raise NotImplementedError


    elif lang_pair == 'th-vi':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        source_loader = load_data_th
        if source_type == 'w':
            try:
                train_data, val_data, inp_dict = \
                    loader(th_vi['train_loaded_th'],
                           th_vi['test_loaded_th'],
                           source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(th_vi['train_raw_th'],
                          th_vi['test_raw_th'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_data, val_data,
                      th_vi['train_loaded_th'],
                      th_vi['test_loaded_th'])
        elif source_type == 'c':
            train_data, val_data, inp_dict = \
                maker(th_vi['train_raw_th'],
                      th_vi['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'tcc':
            train_data, val_data, inp_dict = \
                maker(th_vi['train_raw_th'],
                      th_vi['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(th_vi['train_raw_th'],
                      th_vi['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'wp':
            train_data, val_data, inp_dict = \
                maker(th_vi['train_raw_th'],
                      th_vi['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        source_loader = load_data_vi
        if tgt_type == 'w':
            try:
                train_target, val_target, tgt_dict = \
                    loader(th_vi['train_loaded_vi'],
                           th_vi['test_loaded_vi'],
                       tgt_dict_path)
            except:
                train_target, val_target, tgt_dict = \
                    maker(th_vi['train_raw_vi'],
                          th_vi['test_raw_vi'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_target, val_target,
                      th_vi['train_loaded_vi'],
                      th_vi['test_loaded_vi'])
        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(th_vi['train_raw_vi'],
                      th_vi['test_raw_vi'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)

    elif lang_pair == 'vi-en':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        source_loader = load_data_vi
        if source_type == 'w':
            try:
                train_data, val_data, inp_dict = \
                    loader(vi_en['train_loaded_vi'],
                           vi_en['test_loaded_vi'],
                           source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(vi_en['train_raw_vi'],
                          vi_en['test_raw_vi'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_data, val_data,
                      vi_en['train_loaded_vi'],
                      vi_en['test_loaded_vi'])
        elif source_type == 'c':
            maker(vi_en['train_raw_vi'],
                  vi_en['test_raw_vi'],
                  source_loader, lang_pair, vocab_type=source_type, source=source)
        elif source_type == 'bpe':
            maker(vi_en['train_raw_vi'],
                  vi_en['test_raw_vi'],
                  source_loader, lang_pair, vocab_type=source_type, source=source)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        source_loader = load_data_en
        if tgt_type == 'w':
            try:
                train_target, val_target, tgt_dict = \
                    loader(vi_en['train_loaded_en'],
                           vi_en['test_loaded_en'],
                       tgt_dict_path)
            except:
                train_target, val_target, tgt_dict = \
                    maker(vi_en['train_raw_en'],
                          vi_en['test_raw_en'],
                          source_loader, lang_pair, vocab_type=source_type, source=source)
                saver(train_target, val_target,
                      vi_en['train_loaded_en'],
                      vi_en['test_loaded_en'])

        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(vi_en['train_raw_en'],
                      vi_en['test_raw_en'],
                      source_loader, lang_pair, vocab_type=source_type, source=source)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print("load complete, size of train data and target", len(train_data), len(train_target))
    return train_data, train_target, val_data, val_target, inp_dict, tgt_dict


def load_test_data(lang_pair, source_type, tgt_type):

    return val_data, val_target, inp_dict, tgt_dict