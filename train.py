import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import random

import time
import math
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from subprocess import check_call

cudafloat = torch.cuda.FloatTensor
cudalong = torch.cuda.LongTensor
import os

from utils import *
from model import Encoder, CharEncoder, LuongDecoder
from load_data import load_data
import torch.nn.utils.rnn as rnn_utils

#############################
### Config and Parameters ###
############################

''' continue '''
load_model = False

''' seg or sent mode'''
seg = False

''' language pair  '''
source_lang = 'zh'
tgt_lang = 'en'
lang_pair = source_lang + '-' + tgt_lang

# model, choose between c - char,  w - word, bpe - byte-pair encoding, wp - wordpiece
source_type = 'w'
tgt_type = 'w'
cnn = True
mode = source_type + '2' + tgt_type  # w2w, w2c, c2c

''' optim '''
learning_rate = 1e-4
dropout = 0.2
grad_clip = 2
N = 64

''' Encoder config '''
embed_dim = 256 if source_type == 'c' else 512
poolstride = 5
en_bi = False
en_layers = 1
en_H = 512

if source_type == 'tcc' and cnn is True:
    k_num = [300, 300, 350, 350]
    k_size = [1, 2, 3, 4]
    highway_layers = 4

else:
    k_num = [200, 200, 250, 250, 300, 300, 300, 300]
    # k_num = [300, 300, 350, 350, 400, 400, 400, 400]
    k_size = [1, 2, 3, 4, 5, 6, 7, 8]
    highway_layers = 1

''' Decoder config '''
de_embed = 512
de_H = 512
de_layers = 1
de_bi = False
en_Hbi = en_H * (2 if en_bi == True else 1)

''' File path '''

th_en_ref = {
    'sent':'th-en_sent/ted_test_th-en.en.tok2',
    'seg':'ted_test_th-en.en.tok_seg',
}

th_vi_ref = "th-vi/ted_test_th-vi.vi.tok"
vi_en_ref = "vi-en_sent/ted_test_vi-en.en.tok"
ko_en_ref = "ko-en/ted_test_ko-en.en.tok"

print("config loaded: lang_pair: %s, model: %s" % (lang_pair, mode))
###########################
### Load Data and Dict ####
###########################

train_data, train_target, val_data, val_target, inp_dict, tgt_dict = load_data(lang_pair, source_type, tgt_type)
inp_sz = len(inp_dict)
out_sz = len(tgt_dict)
inp_dict_i2c = {v: k for k, v in inp_dict.items()}
tgt_dict_i2c = {v: k for k, v in tgt_dict.items()}

print("size of val data and val target", len(val_data), len(val_target))

print("sample input 0", train_data[0])
print("sample output 0", train_target[0])
print("sample input -4", train_data[-4])
print("sample output -4", train_target[-4])
print("sample input -2", train_data[-2])
print("sample output -2", train_target[-2])
print("sample input -3", train_data[-3])
print("sample output -3", train_target[-3])
print("sample val data 1", val_data[0])
print("sample val target 1", val_target[0])
print("sample val data last", val_data[1])
print("sample val target last", val_target[1])
print("sample val data last", val_data[-1])
print("sample val target last", val_target[-1])

print(inp_dict['ZERO'])
print(inp_dict['UNK'])
print(inp_dict['SOS'])
print(inp_dict['EOS'])
print(tgt_dict['ZERO'])
print(tgt_dict['UNK'])
print(tgt_dict['SOS'])
print(tgt_dict['EOS'])
# print("inp dict index 0-3", inp_dict_i2c[0], inp_dict_i2c[1], inp_dict_i2c[2], inp_dict_i2c[3])
# print("tgt dict index 0-3", tgt_dict_i2c[0], tgt_dict_i2c[1], tgt_dict_i2c[2], tgt_dict_i2c[3])


# filter out very long sequence
train_data_fil, train_target_fil = filter_data_len(train_data, train_target, mode=mode)

print("size of train data, before and after filter", len(train_data), len(train_data_fil))
print("max len of val data and tgt_type", seq_len_finder(val_data), seq_len_finder(val_target))

train_x, train_y = sent2vec(train_data_fil, train_target_fil, inp_dict, tgt_dict)
val_x, val_y = sent2vec(val_data, val_target, inp_dict, tgt_dict)

print("sample data after vectorizing", train_x[0])

#############
### Train ###
#############

if cnn is False :
    encoder = Encoder(num_embed=inp_sz, embed_dim=embed_dim, N=N, dropout=dropout, k_num=k_num, k_size=k_size, poolstride=poolstride,
                      en_bi=en_bi, en_layers=en_layers, en_H=en_H)
elif cnn is True:
    encoder = CharEncoder(num_embed=inp_sz, embed_dim=embed_dim, N=N, dropout=dropout, k_num=k_num, k_size=k_size,
                          poolstride=poolstride,en_bi=en_bi, en_layers=en_layers, en_H=en_H, highway_layers=highway_layers)
else:
    NotImplementedError

print(encoder)

decoder = LuongDecoder(de_embed=de_embed, de_H=de_H, en_Hbi=en_Hbi, de_layers=de_layers, dropout=dropout, de_bi=de_bi, N=N, out_sz=out_sz)

if load_model is True:
    encoder.load_state_dict(torch.load('last_encoder_weight_vi-en'))
    decoder.load_state_dict(torch.load('last_decoder_weight_vi-en'))
    print("last model loaded")

decoder.cuda()
encoder.cuda()
en_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
de_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(size_average=False, ignore_index=0) # sums loss, ignores loss if target is zero

graph_train, graph_val = [], []
best_val_loss = 100.0
best_bleu_score = 0.0
best_val_acc = 0.0
n_epochs = 500
update_size = 200
train_remainder = len(train_x) % N
val_remainder = len(val_x) % N

for epoch in range(n_epochs):
    start_time = time.time()
    train_loss, train_acc = 0.0, 0.0
    val_loss, val_acc = 0.0, 0.0
    correct = 0
    total_loss = 0
    total_val_len = 0
    en_hidden = encoder.init_hidden()
    de_hidden = decoder.init_hidden()

    encoder.train()
    decoder.train()
    for batch in range(update_size):
        loss = 0
        x_batch, y_batch = [], []
        for i in range(N):
            rand = random.randrange(0, len(train_x))
            x_batch.append(train_x[rand])
            y_batch.append(train_y[rand])

        # rand = random.randrange(0, len(train_x) - train_remainder - N)
        # x_batch = train_x[rand:rand + N]
        # y_batch = train_y[rand:rand + N]

        data, target, len_xs, len_ys, train_idx = vec2tensor(x_batch, y_batch)

        # print(data, target)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        en_hidden = repackage_hidden(en_hidden)
        de_hidden = repackage_hidden(de_hidden)

        # forward, backward, optimize
        en_out, en_hidden = encoder(data, en_hidden, len_xs)

        if cnn is False:
            x_mask_tensor = mask_matrix(len_xs)
        else:
            x_mask_tensor = mask_cnn(len_xs, poolstride)

        de_hidden = en_hidden
        de_in = Variable(torch.zeros(decoder.N, 1).type(cudalong))
        de_in.data.fill_(2)

        for di in range(max(len_ys)):
            t_step = di+1

            # teacher forcing
            # y_mask_list = [1 if t_step <= len_y else 0 for len_y in len_ys]
            # de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden, y_mask_list, x_mask_tensor)  # (W=1,N,Out)

            de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden, x_mask_tensor)  # (W=1,N,Out)
            de_out = de_out.squeeze(0)  # (N, Out)
            # print(de_out)
            target_T = target.transpose(0, 1)  # (N,W) => (W, N)
            loss += criterion(de_out, target_T[di])  # (N,Out) and (N)

            # de_in = Variable(de_out.data.max(1)[1].unsqueeze(1).type(cudalong))
            # teacher forcing
            de_in = target_T[di].unsqueeze(1)

        train_loss += loss.data[0] / sum(len_ys)
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
        en_optimizer.step()
        de_optimizer.step()

    # evaluate with validation set
    encoder.eval()
    decoder.eval()
    en_hidden_val = encoder.init_hidden()
    de_hidden_val = decoder.init_hidden()

    with open('hypothesis.txt', 'w', encoding='utf-8') as translated_val:
        with open('hypothesis_raw.txt', 'w', encoding='utf-8') as raw_translated_val:
            total_trans_list = []
            for batch in range(0, len(val_x) - val_remainder, N):
                trans_list = [[] for _ in range(N)]
                raw_list = [[] for _ in range(N)]
                xval_batch = val_x[batch:batch + N]
                yval_batch = val_y[batch:batch + N]
                data_val, target_val, len_xsval, len_ysval, val_idx = vec2tensor(xval_batch, yval_batch)

                # print("val vec", xval_batch[0])
                # print("val data", data_val[0])
                # print("val target", target_val[0])
                # print("val idx", val_idx[0])

                data_val, target_val = data_val.cuda(), target_val.cuda()
                data_val, target_val = Variable(data_val, volatile=True), Variable(target_val, volatile=True)
                en_hidden_val = repackage_hidden(en_hidden_val)
                de_hidden_val = repackage_hidden(de_hidden_val)

                en_out_val, en_hidden_val = encoder(data_val, en_hidden_val, len_xsval)
                de_hidden_val = en_hidden_val

                if cnn is False:
                    xval_mask_tensor = mask_matrix(len_xsval)
                else:
                    xval_mask_tensor = mask_cnn(len_xsval, poolstride)

                de_in_val = Variable(torch.zeros(decoder.N, 1).type(cudalong))
                de_in_val.data.fill_(2)

                for di in range(max(len_ysval)):
                    t_step = di + 1
                    # yval_mask_list = [1 if t_step <= len_y else 0 for len_y in len_ysval]
                    # de_out_val, de_hidden_val, attn = decoder(de_in_val, en_out_val, de_hidden_val, yval_mask_list, xval_mask_tensor)  # (W=1,N,Out)
                    de_out_val, de_hidden_val, attn = decoder(de_in_val, en_out_val, de_hidden_val, xval_mask_tensor)  # (W=1,N,Out)
                    de_out_val = de_out_val.squeeze(0)  # (N, Out)
                    target_val_T = target_val.transpose(0, 1)  # (N,W) => (W, N)
                    loss = criterion(de_out_val, target_val_T[di])  # (N,Out) and (N)
                    val_loss += loss.data[0]

                    de_in_val = Variable(de_out_val.data.max(1)[1].unsqueeze(1).type(cudalong), volatile=True)
                    pred_val = de_out_val.data.max(1)[1].squeeze().contiguous()  # get the index of the max log-probability
                    target_pred = target_val_T[di].contiguous()
                    correct += pred_val.eq(target_pred.data.view_as(pred_val)).cpu().sum()

                    for i in range(N):
                        word_vec = pred_val.view(-1)[i]
                        word = tgt_dict_i2c[word_vec]
                        trans_list[i].append(word)
                        raw_list[i].append(word_vec)

                val_loss /= sum(len_ysval)
                total_val_len += sum(len_ysval)

                _, unsorted_trans = zip(*sorted(zip(val_idx, trans_list)))
                _, unsorted_raw = zip(*sorted(zip(val_idx, raw_list)))

                # print("unsorted raw", unsorted_raw[0])
                # print("unsorted trans", unsorted_trans[0])
                # print("idx", _[0])

                # unsorted_trans = sorted(trans_list, key=val_idx)
                for trans in unsorted_trans:
                    word_list = []
                    for word in trans:
                        if word in ('EOS', 'ZERO'):
                            break
                        word_list.append(word)
                    if tgt_type == 'c':
                        sent = ''.join(word_list)
                    else:
                        sent = ' '.join(word_list)
                    if tgt_type == 'bpe':
                        sent = multireplace(sent, {'@@ ':''})
                    translated_val.write(sent + '\n')

                for raw in unsorted_raw:
                    sent = ' '.join([str(i) for i in raw])
                    raw_translated_val.write(sent+ '\n')


    train_loss /= len(train_data)
    val_loss /= len(val_data)

    graph_train.append(train_loss)
    graph_val.append(val_loss)
    val_acc = correct / (total_val_len * N)

    ## calculate BLEU

    bleu_score = 0

    with open('hypothesis.txt', 'r') as raw_file, open('hypothesis.txt.norm', 'w') as normalized_file:
        check_call(["perl", "moses/tokenizer/normalize-punctuation.perl", "-l en"],
                   stdin=raw_file, stdout=normalized_file)

    with open('hypothesis.txt.norm', 'r') as normalized_file, open('hypothesis.txt.tok', 'w') as tokenized_file:
        check_call(["perl", "moses/tokenizer/tokenizer.perl", "-l en"],
                   stdin=normalized_file, stdout=tokenized_file)

    if lang_pair == 'th-en' and (seg == False):
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_en_ref['sent']],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair == 'th-en' and (seg == True):
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_en_ref['seg']],
                       stdin=input_file, stdout=bleu_file)

    elif lang_pair == 'th-vi':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_vi_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair == 'vi-en':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", vi_en_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair == 'ko-en':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", ko_en_ref],
                       stdin=input_file, stdout=bleu_file)
    else:
        raise ValueError



    with open('bleu_score.txt', 'r') as h:
        h = h.read()
        bleu = ''+ h[7] + h[8] + h[9] + h[10]
        bleu = float(bleu)
        bleu_score = bleu

    print('[%d] train loss: %.3f val loss: %.4f acc: %.3f time: %.3f bleu: %.3f'  % \
          (epoch + 1, train_loss, val_loss, val_acc,
           time.time() - start_time, bleu_score))

    # torch.save(encoder.state_dict(), 'last_encoder_weight_%s' % lang_pair)
    # torch.save(decoder.state_dict(), 'last_decoder_weight_%s' % lang_pair)

    # normally save for best loss but in this case not so indicative.
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(encoder.state_dict(), 'best_loss_encoder_weight_%s-%s' % (lang_pair, mode))
    #     torch.save(decoder.state_dict(), 'best_loss_decoder_weight_%s-%s' % (lang_pair, mode))
    #     print('saving least val loss model from epoch [%d]' % (epoch + 1))

    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        # torch.save(encoder.state_dict(), 'best_acc_encoder_weight_%s-%s' % (lang_pair, mode))
        # torch.save(decoder.state_dict(), 'best_acc_decoder_weight_%s-%s' % (lang_pair, mode))
        print('saving most bleu acc  model from epoch [%d]' % (epoch + 1))
