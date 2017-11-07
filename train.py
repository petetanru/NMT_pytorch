import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data

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

#############################
### Config and Parameters ###
############################


''' continue '''
load_model = False

''' language pair  '''
source_lang = 'th'
tgt_lang = 'vi'
lang_pair = source_lang + '-' + tgt_lang

# model, choose between c - char,  w - word, bpe - byte-pair encoding, wp - wordpiece
source_type = 'c'
tgt_type = 'w'
cnn = True
mode = source_type + '2' + tgt_type  # w2w, w2c, c2c

''' optim '''
learning_rate = 1e-4
dropout = 0.2
grad_clip = 1
N = 256

''' Encoder config '''
embed_dim = 128 if source_type == 'c' else 256
poolstride = 5
en_bi = False
en_layers = 1
en_H = 256

if source_type == 'tcc' and cnn is True:
    k_num = [300, 300, 350, 350]
    k_size = [1, 2, 3, 4]

else:
    k_num = [200, 200, 250, 250, 300, 300, 300, 300]
    # k_num = [300, 300, 350, 350, 400, 400, 400, 400]
    k_size = [1, 2, 3, 4, 5, 6, 7, 8]

''' Decoder config '''
de_embed = 256
de_H = 256
de_layers = 1
de_bi = False
en_Hbi = en_H * (2 if en_bi == True else 1)

''' File path '''
th_en_ref = "th-en/ted_test_th-en.en.tok_seg"
th_vi_ref = "th-vi/ted_test_th-vi.vi.tok"
vi_en_ref = "vi-en/ted_test_vi-en.en.tok"

print("config loaded: lang_pair: %s, model: %s" % (lang_pair, mode))
###########################
### Load Data and Dict ####
###########################

train_data, train_target, val_data, val_target, inp_dict, tgt_dict = load_data(lang_pair, source_type, tgt_type)

tgt_dict_i2c = {v: k for k, v in tgt_dict.items()}
inp_sz = len(inp_dict)
out_sz = len(tgt_dict)
print("size of inp and out dict", inp_sz, out_sz)
print("sample input", train_data[0])
print("sample output", train_target[0])
print("sample val data 1", val_data[0])
print("sample val target 1", val_target[0])
print("sample val data 2", val_data[11])
print("sample val target 2", val_target[11])

train_data_fil, train_target_fil = filter_data_len(train_data, train_target, mode=mode)
# val_data_fil, val_target_fil = filter_data_len(val_data, val_target, mode=mode)

print("size of train data, before and after filter", len(train_data), len(train_data_fil))
# print("seize of val data, before and after filter", len(val_data), len(val_data_fil))
print("len of val data and tgt_type", seq_len_finder(val_data), seq_len_finder(val_target))

train_data, train_target = train_data_fil, train_target_fil
# val_data, val_target = val_data_fil, val_target_fil

#############
### Train ###
#############

if cnn is False :
    encoder = Encoder(num_embed=inp_sz, embed_dim=embed_dim, N=N, dropout=dropout, k_num=k_num, k_size=k_size, poolstride=poolstride,
                      en_bi=en_bi, en_layers=en_layers, en_H=en_H)
elif cnn is True:
    encoder = CharEncoder(num_embed=inp_sz, embed_dim=embed_dim, N=N, dropout=dropout, k_num=k_num, k_size=k_size,
                          poolstride=poolstride,en_bi=en_bi, en_layers=en_layers, en_H=en_H)
else:
    NotImplementedError

print(encoder)

decoder = LuongDecoder(de_embed=de_embed, de_H=de_H, en_Hbi=en_Hbi, de_layers=de_layers, dropout=dropout, de_bi=de_bi, N=N,
                       out_sz=out_sz)

if load_model is True:
    encoder.load_state_dict(torch.load('last_encoder_weight_th-en'))
    decoder.load_state_dict(torch.load('last_decoder_weight_th-en'))
    print("last model loaded")

decoder.cuda()
encoder.cuda()
en_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
de_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(size_average=True)

graph_train, graph_val = [], []
best_val_loss = 100.0
best_bleu_score = 0.0
best_val_acc = 0.0
n_epochs = 200
train_remainder = len(train_data) % N
val_remainder = len(val_data) % N

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
    for batch in range(0, len(train_data) - train_remainder, N):
        loss = 0
        data_raw = train_data[batch:batch + N]
        target_raw = train_target[batch:batch + N]
        data, target, _seq_len_x, seq_len_y = train_vectorize(data_raw, target_raw, inp_dict, tgt_dict, mode=mode)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        en_hidden = repackage_hidden(en_hidden)
        de_hidden = repackage_hidden(de_hidden)

        # forward, backward, optimize
        en_out, en_hidden = encoder(data, en_hidden)
        de_hidden = en_hidden
        de_in = Variable(torch.zeros(decoder.N, 1).type(cudalong))

        for di in range(seq_len_y):
            de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden)  # (W=1,N,Out)
            de_out = de_out.squeeze(0)  # (N, Out)
            target_T = target.transpose(0, 1)  # (N,W) => (W, N)
            loss += criterion(de_out, target_T[di])  # (N,Out) and (N)

            de_in = target_T[di].unsqueeze(1)
        train_loss += loss.data[0] / seq_len_y
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        loss.backward(retain_variables=True)
        torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
        en_optimizer.step()
        de_optimizer.step()

    # evaluate with validation set
    encoder.eval()
    decoder.eval()

    with open('hypothesis.txt', 'w', encoding='utf-8') as translated_val:
        total_trans_list = []
        for batch in range(0, len(val_data) - val_remainder, N):
            trans_list = [[] for _ in range(N)]
            data_raw = val_data[batch:batch + N]
            target_raw = val_target[batch:batch + N]
            data, target, _seq_len_x, seq_len_y = train_vectorize(data_raw, target_raw, inp_dict, tgt_dict, mode=mode)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            en_hidden = repackage_hidden(en_hidden)
            de_hidden = repackage_hidden(de_hidden)

            en_out, en_hidden = encoder(data, en_hidden)
            de_hidden = en_hidden
            de_in = Variable(torch.zeros(decoder.N, 1).type(cudalong))

            for di in range(seq_len_y):
                de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden)  # (W=1,N,Out)
                de_out = de_out.squeeze(0)  # (N, Out)
                target_T = target.transpose(0, 1)  # (N,W) => (W, N)
                loss = criterion(de_out, target_T[di])  # (N,Out) and (N)
                val_loss += loss.data[0] / (seq_len_y)

                de_in = Variable(de_out.data.max(1)[1].type(cudalong), volatile=True)

                pred = de_out.data.max(1)[1].squeeze().contiguous()  # get the index of the max log-probability
                target_pred = target_T[di].contiguous()
                correct += pred.eq(target_pred.data.view_as(pred)).cpu().sum()

                for i in range(N):
                    word = tgt_dict_i2c[pred.view(-1)[i]]
                    trans_list[i].append(word)

            total_val_len += seq_len_y

            for trans in trans_list:
                if tgt_type == 'w':
                    sent = ' '.join([word for word in trans if word != 'EOS'])
                elif tgt_type == 'c':
                    sent = ''.join([char for char in trans if char != 'EOS'])
                translated_val.write(sent + '\n')


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

    if lang_pair == 'th-en':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_en_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair == 'th-vi':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_vi_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair == 'vi-en':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", vi_en_ref],
                       stdin=input_file, stdout=bleu_file)
    else:
        raise ValueError

    with open('bleu_score.txt', 'r') as h:
        h = h.read()
        bleu = ''+ h[7] + h[8] + h[9] + h[10]
        bleu = float(bleu)
        print(bleu)
        bleu_score = bleu

    print('[%d] train loss: %.3f val loss: %.4f acc: %.3f time: %.3f bleu: %.3f'  % \
          (epoch + 1, train_loss, val_loss, val_acc,
           time.time() - start_time, bleu_score))

    torch.save(encoder.state_dict(), 'last_encoder_weight_%s' % lang_pair)
    torch.save(decoder.state_dict(), 'last_decoder_weight_%s' % lang_pair)

    # normally save for best loss but in this case not so indicative.
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(encoder.state_dict(), 'best_loss_encoder_weight_%s-%s' % (lang_pair, mode))
    #     torch.save(decoder.state_dict(), 'best_loss_decoder_weight_%s-%s' % (lang_pair, mode))
    #     print('saving least val loss model from epoch [%d]' % (epoch + 1))

    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        torch.save(encoder.state_dict(), 'best_acc_encoder_weight_%s-%s' % (lang_pair, mode))
        torch.save(decoder.state_dict(), 'best_acc_decoder_weight_%s-%s' % (lang_pair, mode))
        print('saving most bleu acc  model from epoch [%d]' % (epoch + 1))
