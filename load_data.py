import pickle
from utils import *

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





''' code '''

def load_data(lang_pair, source_type, tgt_type):
    if lang_pair == 'th-en':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        if source_type == 'w':
            try:
                with open(th_en['train_loaded_th'], 'rb') as f1:
                    train_data = pickle.load(f1)
                with open(th_en['test_loaded_th'], 'rb') as f2:
                    val_data = pickle.load(f2)
                with open(source_dict_path, 'rb') as f3:
                    inp_dict = pickle.load(f3)
            except:
                train_data, inp_dict = load_data_th(th_en['train_raw_th'], lang_pair,
                                                    vocab_type = source_type, source=source, train=True)
                val_data = load_data_th(th_en['test_raw_th'], lang_pair, vocab_type =source_type,
                                        source=source, train=False)
                with open(th_en['train_loaded_th'], 'wb') as f4:
                    pickle.dump(train_data, f4)
                with open(th_en['test_loaded_th'], 'wb') as f5:
                    pickle.dump(val_data, f5)
                # dict already saved from load data fnc

        elif source_type == 'c':
            train_data, inp_dict = load_data_th(th_en['train_raw_th'], lang_pair, vocab_type=source_type,
                                                source=source, train=True)
            val_data = load_data_th(th_en['test_raw_th'], lang_pair, vocab_type=source_type, source=source, train=False)

        elif source_type == 'tcc':
            train_data, inp_dict = load_data_th(th_en['train_raw_th'], lang_pair, vocab_type=source_type,
                                                source=source, train=True)
            val_data = load_data_th(th_en['test_raw_th'], lang_pair, vocab_type=source_type, source=source, train=False)

        elif source_type == 'bpe':
            train_data, inp_dict = load_data_th(th_en['train_bpe_th'], lang_pair, vocab_type=source_type, source=source,
                                                train=True)
            val_data = load_data_th(th_en['test_bpe_th'], lang_pair, vocab_type=source_type, source=source, train=False)

        elif source_type == 'wp':
            load_data_th(th_en['train_raw_th'], lang_pair, vocab_type=source_type, source=source, train=True)
            train_data, inp_dict = load_data_th(th_en['train_raw_th'], lang_pair, vocab_type=source_type, source=source, train=True)
            val_data = load_data_th(th_en['test_raw_th'], lang_pair, vocab_type=source_type, source=source, train=False)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[2]+lang_pair[3]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        if tgt_type == 'w':
            try:
                with open(th_en['train_loaded_en'], 'rb') as f1:
                    train_target = pickle.load(f1)
                with open(th_en['test_loaded_en'], 'rb') as f3:
                    val_target = pickle.load(f3)
                with open(tgt_dict_path, 'rb') as f6:
                    tgt_dict = pickle.load(f6)

            except:
                train_target, tgt_dict = load_data_en(th_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                                           source=source, train=True)
                val_target = load_data_en(th_en['test_raw_en'], lang_pair, vocab_type=tgt_type,
                                               source=source, train=False)
                with open(th_en['train_loaded_en'], 'wb') as f1:
                    pickle.dump(train_target, f1)
                with open(th_en['test_loaded_en'], 'wb') as f3:
                    pickle.dump(val_target, f3)

        elif tgt_type == 'c':
            train_target, tgt_dict = load_data_en(th_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                             source=source, train=True)
            val_target = load_data_en(th_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                            source=source, train=False)
    elif lang_pair == 'th-vi':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        if source_type == 'w':
            try: # load data and dict
                with open(th_vi['train_loaded_th'], 'rb') as f1:
                    train_data = pickle.load(f1)
                with open(source_dict_path, 'rb') as f2:
                    inp_dict = pickle.load(f2)
            except: # save data and dict
                train_data, inp_dict = load_data_th(th_vi['train_raw_th'], lang_pair,
                                                    vocab_type=source_type, source=source, train=True)
                with open(th_vi['train_loaded_th'], 'wb') as f3:
                    pickle.dump(train_data, f3)
                #dict already saved in build vocab function
            try:
                with open(th_vi['test_loaded_th'], 'rb') as f5:
                    val_data = pickle.load(f5)
            except:
                val_data = load_data_th(th_vi['test_raw_th'], lang_pair, vocab_type=source_type,
                                        source=source, train=False)
                with open(th_vi['test_loaded_th'], 'wb') as f6:
                    pickle.dump(val_data, f6)

        elif source_type == 'c':
            train_data, inp_dict = load_data_th(th_vi['train_raw_th'], lang_pair, vocab_type=source_type,
                                                source=source, train=True)
            val_data = load_data_th(th_vi['test_raw_th'], lang_pair, vocab_type=source_type, source=source,
                                    train=False)

        elif source_type == 'tcc':
            train_data, inp_dict = load_data_th(th_vi['train_raw_th'], lang_pair, vocab_type=source_type,
                                                source=source, train=True)
            val_data = load_data_th(th_vi['test_raw_th'], lang_pair, vocab_type=source_type, source=source,
                                    train=False)

        elif source_type == 'bpe':
            train_data, inp_dict = load_data_th(th_vi['train_bpe_th'], lang_pair, vocab_type=source_type, source=source, train=True)
            val_data = load_data_th(th_vi['test_bpe_th'], lang_pair, vocab_type=source_type, source=source, train=False)

        elif source_type == 'wp':
            train_data, inp_dict = load_data_th(th_vi['train_wp_th'], lang_pair, vocab_type=source_type, source=source, train=True)
            val_data = load_data_th(th_vi['test_wp_th'], lang_pair, vocab_type=source_type, source=source, train=False)

        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[2]+lang_pair[3]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        if tgt_type == 'w':
            try:
                with open(th_vi['train_loaded_vi'], 'rb') as f1:
                    train_target = pickle.load(f1)
                with open(th_vi['test_loaded_vi'], 'rb') as f3:
                    val_target = pickle.load(f3)
                with open(tgt_dict_path, 'rb') as f6:
                    tgt_dict = pickle.load(f6)

            except:
                train_target, tgt_dict = load_data_vi(th_vi['train_raw_vi'], lang_pair, vocab_type=tgt_type, source=source, train=True)
                val_target = load_data_vi(th_vi['test_raw_vi'], lang_pair, vocab_type=tgt_type,source=source, train=False)
                with open(th_vi['train_loaded_vi'], 'wb') as f1:
                    pickle.dump(train_target, f1)
                with open(th_vi['test_loaded_vi'], 'wb') as f3:
                    pickle.dump(val_target, f3)

        elif tgt_type == 'c':
            train_target, tgt_dict = load_data_vi(th_vi['train_raw_vi'], lang_pair, vocab_type=tgt_type,
                                                       source=source, train=True)
            val_target = load_data_vi_char(th_vi['test_raw_vi'], lang_pair, vocab_type=tgt_type,
                                           source=source, train=False)
    elif lang_pair == 'vi-en':
        ''' source '''
        source = True
        source_lang = lang_pair[0]+lang_pair[1]
        source_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, source_type, 'src', source_lang)
        if source_type == 'w':
            try:
                with open(vi_en['train_loaded_vi'], 'rb') as f1:
                    train_data = pickle.load(f1)
                with open(vi_en['test_loaded_vi'], 'rb') as f2:
                    val_data = pickle.load(f2)
                with open(source_dict_path, 'rb') as f3:
                    inp_dict = pickle.load(f3)
            except:
                train_data, inp_dict = load_data_vi(vi_en['train_raw_vi'], lang_pair,
                                                    vocab_type = source_type, source=source, train=True)
                val_data = load_data_vi(vi_en['test_raw_vi'], lang_pair, vocab_type =source_type,
                                        source=source, train=False)
                with open(vi_en['train_loaded_vi'], 'wb') as f4:
                    pickle.dump(train_data, f4)
                with open(vi_en['test_loaded_vi'], 'wb') as f5:
                    pickle.dump(val_data, f5)
                # dict already saved from load data fnc

        elif source_type == 'c':
            train_data, inp_dict = load_data_vi(vi_en['train_raw_vi'], lang_pair, vocab_type=source_type,
                                                source=source, train=True)
            val_data = load_data_vi(vi_en['test_raw_vi'], lang_pair, vocab_type=source_type, source=source, train=False)

        elif source_type == 'bpe':
            train_data, inp_dict = load_data_vi(vi_en['train_bpe_vi'], lang_pair, vocab_type=source_type, source=source, train=True)
            val_data = load_data_vi(vi_en['test_bpe_vi'], lang_pair, vocab_type=source_type, source=source, train=False)
        else:
            raise NotImplementedError

        ''' target '''
        source = False
        tgt_lang = lang_pair[2]+lang_pair[3]
        tgt_dict_path = '%s/%s.%s.%s.%s.pkl' % (lang_pair, lang_pair, tgt_type, 'tgt', tgt_lang)
        if tgt_type == 'w':
            try:
                with open(vi_en['train_loaded_en'], 'rb') as f1:
                    train_target = pickle.load(f1)
                with open(vi_en['test_loaded_en'], 'rb') as f3:
                    val_target = pickle.load(f3)
                with open(tgt_dict_path, 'rb') as f6:
                    tgt_dict = pickle.load(f6)

            except:
                train_target, tgt_dict = load_data_en(vi_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                                           source=source, train=True)
                val_target = load_data_en(vi_en['test_raw_en'], lang_pair, vocab_type=tgt_type,
                                               source=source, train=False)
                with open(vi_en['train_loaded_en'], 'wb') as f1:
                    pickle.dump(train_target, f1)
                with open(vi_en['test_loaded_en'], 'wb') as f3:
                    pickle.dump(val_target, f3)

        elif tgt_type == 'c':
            train_target, tgt_dict = load_data_en(vi_en['train_raw_en'], lang_pair, vocab_type=tgt_type,
                                             source=source, train=True)
            val_target = load_data_en(vi_en['test_raw_en'], lang_pair, vocab_type=tgt_type,
                                            source=source, train=False)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print("load complete, size of train data and target", len(train_data), len(train_target))
    return train_data, train_target, val_data, val_target, inp_dict, tgt_dict