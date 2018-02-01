import pickle
import os
from utils import build_vocab, en_unicodeToAscii
import nltk
from tcc import tcc
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
    'train_bpe_th' : 'th-en_sent/th-en_train_bpe_ready.th.bpe',
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

# segment for vi_en
# vi_en ={
#     'train_raw_vi': 'vi-en/ted_train_vi-en.vi',
#     'train_raw_en': 'vi-en/ted_train_vi-en.en',
#     'test_raw_vi': 'vi-en/ted_test_vi-en.vi',
#     'test_raw_en': 'vi-en/ted_test_vi-en.en',
#     'train_loaded_vi': 'vi-en/ted_train_vi-en.vi.pkl',
#     'train_loaded_en': 'vi-en/ted_train_vi-en.en.pkl',
#     'test_loaded_vi': 'vi-en/ted_test_vi-en.vi.pkl',
#     'test_loaded_en': 'vi-en/test_test_vi-en.en.pkl',
#     'train_bpe_vi': 'vi-en/ted_train_vi-en.vi.bpe60000.experiment',
#     'test_bpe_vi': 'vi-en/ted_test_vi-en.vi.bpe60000.experiment'
# }

vi_en ={
    'train_raw_vi': 'vi-en_sent/ted_train_vi-en.vi',
    'train_raw_en': 'vi-en_sent/ted_train_vi-en.en',
    'test_raw_vi': 'vi-en_sent/ted_test_vi-en.vi',
    'test_raw_en': 'vi-en_sent/ted_test_vi-en.en',
    'train_loaded_vi': 'vi-en_sent/ted_train_vi-en.vi.pkl',
    'train_loaded_en': 'vi-en_sent/ted_train_vi-en.en.pkl',
    'test_loaded_vi': 'vi-en_sent/ted_test_vi-en.vi.pkl',
    'test_loaded_en': 'vi-en_sent/test_test_vi-en.en.pkl',
    'train_bpe_vi': 'vi-en_sent/ted_train_vi-en.vi.bpe',
    'test_bpe_vi': 'vi-en_sent/ted_test_vi-en.vi.bpe',
    'train_bpe_en': 'vi-en_sent/ted_train_vi-en.en.bpe',
    'test_bpe_en': 'vi-en_sent/ted_test_vi-en.en.bpe',
}

zh_cn_en ={
    'train_raw_zh-cn':'zh-en/ted_train_zh-cn-en.zh-cn',
    'train_tok_zh-cn': 'zh-en/ted_train_zh-cn-en.zh-cn.tokenized.pku',
    'train_raw_en': 'zh-en/ted_train_zh-cn-en.en',
    'test_raw_zh-cn':'zh-en/ted_train_zh-cn-en.zh-cn',
    'test_tok_zh-cn': 'zh-en/ted_test_zh-cn-en.zh-cn.tokenized.pku',
    'test_raw_en': 'zh-en/ted_test_zh-cn-en.en',
    'train_loaded_zh-cn': 'zh-en/ted_train_zh-cn-en.zh-cn.pkl',
    'train_loaded_en': 'zh-en/ted_train_zh-cn-en.en.pkl',
    'test_loaded_zh-cn': 'zh-en/ted_test_zh-cn-en.zh-cn.pkl',
    'test_loaded_en': 'zh-en/test_test_zh-cn-en.en.pkl',
    'train_bpe_zh-cn': 'zh-en/ted_train_zh-cn-en.zh-cn.bpe',
    'test_bpe_zh-cn': 'zh-en/ted_test_zh-cn-en.zh-cn.bpe',
    'train_bpe_en': 'zh-en/ted_train_zh-cn-en.en.bpe',
    'test_bpe_en': 'zh-en/ted_test_zh-cn-en.en.bpe',
}

ko_en ={
    'train_raw_ko': 'ko-en/ted_train_ko-en.ko',
    'train_raw_en': 'ko-en/ted_train_ko-en.en',
    'test_raw_ko': 'ko-en/ted_test_ko-en.ko',
    'test_raw_en': 'ko-en/ted_test_ko-en.en',
    'train_loaded_ko': 'ko-en/ted_train_ko-en.ko.pkl',
    'train_loaded_en': 'ko-en/ted_train_ko-en.en.pkl',
    'test_loaded_ko': 'ko-en/ted_test_ko-en.ko.pkl',
    'test_loaded_en': 'ko-en/test_test_ko-en.en.pkl',
    'train_bpe_ko': 'ko-en/ted_train_ko-en.ko.bpe',
    'test_bpe_ko': 'ko-en/ted_test_ko-en.ko.bpe',
    'train_bpe_en': 'ko-en/ted_train_ko-en.en.bpe',
    'test_bpe_en': 'ko-en/ted_test_ko-en.en.bpe',
}

sent = True

''' language specific loader functions '''
def load_data_th(path, vocab_type):
    output = []
    lang = 'th'
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().splitlines()
        if vocab_type == 'w':
            import deepcut
            for sent in data:
                # sent = th_unicode(sent)  # comment out to allow numbers
                output.append(deepcut.tokenize(sent))
        elif vocab_type == 'c':
            for sent in data:
                # sent = th_unicode(sent)
                output.append(sent)
            print("inside load data", len(output))
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
    return output

def load_data_vi(path, vocab_type):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().splitlines()
        for sent in data:
            if vocab_type == 'w':
                output.append(nltk.word_tokenize(sent))
            elif vocab_type == 'c':
                output.append(sent)
            elif vocab_type == 'bpe':
                word_list = sent.split()
                output.append(word_list)
    return output

def load_data_en(path, vocab_type):
    output = []
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read().splitlines()
        print("printing from loading data. length of en", len(data))
        for sent in data:
            # sent = en_unicodeToAscii(sent)
            if vocab_type == 'w':
                output.append(nltk.word_tokenize(sent))
            elif vocab_type == 'c':
                output.append(sent)
            elif vocab_type == 'bpe':
                word_list = sent.split()
                output.append(word_list)
    return output

def load_data_zh_cn(path, vocab_type):
  output = []
  input_file = os.path.join(path)
  with open(input_file, "r", encoding='utf-8') as f:
    data = f.read().splitlines()
    print("printing from loading data. length of zh", len(data))
    if vocab_type == 'w':
      for sent in data:
          word_list = sent.split()
          output.append(word_list)
    elif vocab_type == 'c':
      for sent in data:
        output.append(sent)
    elif vocab_type == 'bpe':
      for sent in data:
        word_list = sent.split()
        output.append(word_list)
  return output

def load_data_ko(path, vocab_type):
  output = []
  lang = 'ko'
  input_file = os.path.join(path)
  with open(input_file, "r", encoding='utf-8') as f:
    data = f.read().splitlines()
    if vocab_type == 'w':
      from konlpy.tag import Twitter
      twitter = Twitter()
      for sent in data:
        output.append(twitter.morphs(sent))
    elif vocab_type == 'c':
      for sent in data:
        output.append(sent)
    elif vocab_type == 'bpe':
      for sent in data:
        word_list = sent.split()
        output.append(word_list)
  return output


''' general loader, maker, saver functions '''

def loader(train_loaded, val_loaded, dict_src):
    with open(train_loaded, 'rb') as f1:
        train_data = pickle.load(f1)
    with open(val_loaded, 'rb') as f2:
        val_data = pickle.load(f2)
    with open(dict_src, 'rb') as f3:
        inp_dict = pickle.load(f3)
    return train_data, val_data, inp_dict

def maker(train_raw, test_raw, lang_loader, lang_pair, vocab_type, source, curr_lang):
    '''
    :param lang_loader: language specific function
    :return: N-lists of lists of words for train and val data, and dictionary
    '''
    train_data = lang_loader(train_raw, vocab_type)
    val_data = lang_loader(test_raw, vocab_type)
    dict = build_vocab(file=train_data,
                       source=source,
                       vocab_type=vocab_type,
                       lang=curr_lang,
                       lang_pair=lang_pair)
    return train_data, val_data, dict

def saver(train_data, val_data, dict, train_path, val_path, dict_path):
    # saves train, val, and dict to corresponding paths
    with open(train_path, 'wb') as f4:
        pickle.dump(train_data, f4)
    with open(val_path, 'wb') as f5:
        pickle.dump(val_data, f5)
    with open(dict_path, 'wb') as f6:
        pickle.dump(dict, f6)

''' code '''

def load_data(lang_pair, source_type, tgt_type):
    if lang_pair == 'th-en':
        ''' source '''
        source = True
        curr_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', curr_lang)
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
                          lang_loader=source_loader,
                          lang_pair=lang_pair,
                          vocab_type=source_type,
                          source=source,
                          curr_lang=curr_lang)
                saver(train_data, val_data, inp_dict,
                      th_en_sent['train_loaded_th'],
                      th_en_sent['test_loaded_th'],
                      source_dict_path)
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
                          lang_loader=source_loader,
                          lang_pair=lang_pair,
                          vocab_type=source_type,
                          source=source,
                          curr_lang=curr_lang)
                saver(train_data, val_data, inp_dict,
                      th_en['train_loaded_th'],
                      th_en['test_loaded_th'],
                      source_dict_path)
        elif source_type == 'c' and sent is True:
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_raw_th'],
                      th_en_sent['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'c' and sent is False:
            train_data, val_data, inp_dict = \
                maker(th_en['train_raw_th'],
                      th_en['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'tcc':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_raw_th'],
                      th_en_sent['test_raw_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_bpe_th'],
                      th_en_sent['test_bpe_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'wp':
            train_data, val_data, inp_dict = \
                maker(th_en_sent['train_bpe_th'],
                      th_en_sent['test_bpe_th'],
                      source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        curr_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', curr_lang)
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
                          source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
                saver(train_target, val_target, tgt_dict,
                      th_en_sent['train_loaded_en'],
                      th_en_sent['test_loaded_en'],
                      tgt_dict_path)
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
                          source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
                saver(train_target, val_target, tgt_dict,
                      th_en['train_loaded_en'],
                      th_en['test_loaded_en'],
                      tgt_dict_path)
        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(th_en_sent['train_raw_en'],
                      th_en_sent['test_raw_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        elif tgt_type == 'bpe' and sent is True:
            train_target, val_target, tgt_dict = \
                maker(th_en_sent['train_bpe_en'],
                      th_en_sent['test_bpe_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
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
        curr_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', curr_lang)
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
                          source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
                saver(train_data, val_data, inp_dict,
                      vi_en['train_loaded_vi'],
                      vi_en['test_loaded_vi'],
                      source_dict_path)
        elif source_type == 'c':
            train_data, val_data, inp_dict = \
                maker(vi_en['train_raw_vi'],
                  vi_en['test_raw_vi'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(vi_en['train_bpe_vi'],
                  vi_en['test_bpe_vi'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        curr_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', curr_lang)
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
                          source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
                saver(train_target, val_target, tgt_dict,
                      vi_en['train_loaded_en'],
                      vi_en['test_loaded_en'],
                      tgt_dict_path)

        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(vi_en['train_raw_en'],
                      vi_en['test_raw_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        elif tgt_type == 'bpe':
            train_target, val_target, tgt_dict = \
                maker(vi_en['train_bpe_en'],
                      vi_en['test_bpe_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

    elif lang_pair == 'zh-en':
        ''' source '''
        source = True
        curr_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', curr_lang)
        source_loader = load_data_zh_cn
        if source_type == 'w':
            try:
                train_data, val_data, inp_dict = \
                    loader(zh_cn_en['train_loaded_zh-cn'],
                           zh_cn_en['test_loaded_zh-cn'],
                           source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(zh_cn_en['train_tok_zh-cn'],
                          zh_cn_en['test_tok_zh-cn'],
                          source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
                saver(train_data, val_data, inp_dict,
                      zh_cn_en['train_loaded_zh-cn'],
                      zh_cn_en['test_loaded_zh-cn'],
                      source_dict_path)
        elif source_type == 'c':
            train_data, val_data, inp_dict = \
                maker(zh_cn_en['train_raw_zh-cn'],
                      zh_cn_en['test_raw_zh-cn'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(zh_cn_en['train_bpe_zh-cn'],
                      zh_cn_en['test_bpe_zh-cn'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        curr_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', curr_lang)
        source_loader = load_data_en
        if tgt_type == 'w':
            try:
                train_target, val_target, tgt_dict = \
                    loader(zh_cn_en['train_loaded_en'],
                           zh_cn_en['test_loaded_en'],
                            tgt_dict_path)
            except:
                train_target, val_target, tgt_dict = \
                    maker(zh_cn_en['train_raw_en'],
                          zh_cn_en['test_raw_en'],
                          source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
                saver(train_target, val_target, tgt_dict,
                      zh_cn_en['train_loaded_en'],
                      zh_cn_en['test_loaded_en'],
                      tgt_dict_path)

        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(zh_cn_en['train_raw_en'],
                      zh_cn_en['test_raw_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        elif tgt_type == 'bpe':
            train_target, val_target, tgt_dict = \
                maker(zh_cn_en['train_bpe_en'],
                      zh_cn_en['test_bpe_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

    elif lang_pair == 'ko-en':
        ''' source '''
        source = True
        curr_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', curr_lang)
        source_loader = load_data_ko
        if source_type == 'w':
            try:
                train_data, val_data, inp_dict = \
                    loader(ko_en['train_loaded_ko'],
                           ko_en['test_loaded_ko'],
                           source_dict_path)
            except:
                train_data, val_data, inp_dict = \
                    maker(ko_en['train_raw_ko'],
                          ko_en['test_raw_ko'],
                          source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
                saver(train_data, val_data, inp_dict,
                      ko_en['train_loaded_ko'],
                      ko_en['test_loaded_ko'],
                      source_dict_path)
        elif source_type == 'c':
            train_data, val_data, inp_dict = \
                maker(ko_en['train_raw_ko'],
                      ko_en['test_raw_ko'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        elif source_type == 'bpe':
            train_data, val_data, inp_dict = \
                maker(ko_en['train_bpe_ko'],
                      ko_en['test_bpe_ko'],
                  source_loader, lang_pair, vocab_type=source_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        curr_lang = lang_pair[3]+lang_pair[4]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', curr_lang)
        source_loader = load_data_en
        if tgt_type == 'w':
            try:
                train_target, val_target, tgt_dict = \
                    loader(ko_en['train_loaded_en'],
                           ko_en['test_loaded_en'],
                            tgt_dict_path)
            except:
                train_target, val_target, tgt_dict = \
                    maker(ko_en['train_raw_en'],
                          ko_en['test_raw_en'],
                          source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
                saver(train_target, val_target, tgt_dict,
                      ko_en['train_loaded_en'],
                      ko_en['test_loaded_en'],
                      tgt_dict_path)

        elif tgt_type == 'c':
            train_target, val_target, tgt_dict = \
                maker(ko_en['train_raw_en'],
                      ko_en['test_raw_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        elif tgt_type == 'bpe':
            train_target, val_target, tgt_dict = \
                maker(ko_en['train_bpe_en'],
                      ko_en['test_bpe_en'],
                      source_loader, lang_pair, vocab_type=tgt_type, source=source, curr_lang=curr_lang)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print("load complete, size of train data and target", len(train_data), len(train_target))
    print("size of inp and out dict", len(inp_dict), len(tgt_dict))
    return train_data, train_target, val_data, val_target, inp_dict, tgt_dict


def load_test_data(lang_pair, source_type, tgt_type):

    return val_data, val_target, inp_dict, tgt_dict