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

''' language pair(S) '''
source_lang_1 = 'th'
source_lang_2 = 'vi'
tgt_lang = 'en'
lang_pair1 = source_lang_1 + '-' + tgt_lang
lang_pair2 = source_lang_2 + '-' + tgt_lang

# model, choose between c - char,  w - word, bpe - byte-pair encoding
source_type = 'w'
tgt_type = 'w'
cnn = False
mode = source_type + '2' + tgt_type  # w2w, w2c, c2c

''' optim '''
learning_rate = 1e-4
dropout = 0.2
grad_clip = 1
N = 128

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

###########################
### Load Data and Dict ####
###########################

train_data1, train_target1, val_data1, val_target1, inp_dict1, tgt_dict1 = load_data(lang_pair1, source_type, tgt_type)
train_data2, train_target2, val_data2, val_target2, inp_dict2, tgt_dict2 = load_data(lang_pair2, source_type, tgt_type)

# combine dicts
raw_inp_dict = {**inp_dict2, **inp_dict1}
raw_tgt_dict = {**tgt_dict2, **tgt_dict1}

inp_dict, tgt_dict = {}, {}
count, count2 = 0, 0

for k,v in raw_inp_dict.items():
    inp_dict[k] = count
    count += 1
for k2,v2 in raw_tgt_dict.items():
    tgt_dict[k2] = count2
    count2 += 1

google_token = {'<th>':len(inp_dict)+1, '<vi>': len(inp_dict)+2}

inp_dict.update(google_token)

tgt_dict_i2c = {v: k for k, v in tgt_dict.items()}

inp_sz = len(inp_dict)
out_sz = len(tgt_dict)
print("size of inp and out dict", inp_sz, out_sz)
print("sample input 1", train_data1[0])
print("sample output 1", train_target1[0])
print("sample input 2", train_data2[0])
print("sample output 2", train_target2[0])
print("sample val data 1", val_data1[0])
print("sample val target 1", val_target1[0])
print("sample val data 2", val_data2[11])
print("sample val target 2", val_target2[11])

train_data_fil1, train_target_fil1 = filter_data_len(train_data1, train_target1, mode=mode)
train_data_fil2, train_target_fil2 = filter_data_len(train_data2, train_target2, mode=mode)
# val_data_fil, val_target_fil = filter_data_len(val_data, val_target, mode=mode)

print("size of train data, before and after filter 1", len(train_data1), len(train_data_fil1))
print("size of train data, before and after filter 2", len(train_data2), len(train_data_fil2))
# print("seize of val data, before and after filter", len(val_data), len(val_data_fil))
print("len of val data and tgt_type", seq_len_finder(val_data1), seq_len_finder(val_target1))

train_data1, train_target1 = train_data_fil1, train_target_fil1
train_data2, train_target2 = train_data_fil2, train_target_fil2
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
train_remainder1 = len(train_data1) % N
val_remainder1 = len(val_data1) % N

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
    for batch in range(0, len(train_data1) - train_remainder1, N):
        loss = 0

        data_raw1 = train_data1[batch:batch + N]
        data_raw2 = train_data2[batch:batch + N]
        target_raw1 = train_target1[batch:batch + N]
        target_raw2 = train_target2[batch:batch + N]

        ### lang 1

        data1, target1, _seq_len_x1, seq_len_y1 = google_train_vectorize(data_raw1, target_raw1, inp_dict, tgt_dict, lang=source_lang_1, mode=mode)
        data1, target1 = data1.cuda(), target1.cuda()
        data1, target1 = Variable(data1), Variable(target1)
        en_hidden = repackage_hidden(en_hidden)
        de_hidden = repackage_hidden(de_hidden)

        # forward, backward, optimize
        en_out, en_hidden = encoder(data1, en_hidden)
        de_hidden = en_hidden
        de_in = Variable(torch.zeros(decoder.N, 1).type(cudalong))

        for di in range(seq_len_y1):
            de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden)  # (W=1,N,Out)
            de_out = de_out.squeeze(0)  # (N, Out)
            target_T = target1.transpose(0, 1)  # (N,W) => (W, N)
            loss += criterion(de_out, target_T[di])  # (N,Out) and (N)
            de_in = target_T[di].unsqueeze(1)

        ## lang 2

        data2, target2, _seq_len_x, seq_len_y2 = google_train_vectorize(data_raw2, target_raw2, inp_dict, tgt_dict, lang=source_lang_2, mode=mode)
        data2, target2 = data2.cuda(), target2.cuda()
        data2, target2 = Variable(data2), Variable(target2)
        en_hidden = repackage_hidden(en_hidden)
        de_hidden = repackage_hidden(de_hidden)

        # forward, backward, optimize
        en_out, en_hidden = encoder(data2, en_hidden)
        de_hidden = en_hidden
        de_in = Variable(torch.zeros(decoder.N, 1).type(cudalong))

        for di in range(seq_len_y2):
            de_out, de_hidden, attn = decoder(de_in, en_out, de_hidden)  # (W=1,N,Out)
            de_out = de_out.squeeze(0)  # (N, Out)
            target_T = target2.transpose(0, 1)  # (N,W) => (W, N)
            loss += criterion(de_out, target_T[di])  # (N,Out) and (N)
            de_in = target_T[di].unsqueeze(1)


        train_loss += loss.data[0] / (seq_len_y1 + seq_len_y2)
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
        for batch in range(0, len(val_data1) - val_remainder1, N):
            trans_list = [[] for _ in range(N)]
            data_raw = val_data1[batch:batch + N]
            target_raw = val_target1[batch:batch + N]
            data, target, _seq_len_x, seq_len_y = google_train_vectorize(data_raw, target_raw, inp_dict, tgt_dict, lang = source_lang_1, mode=mode)
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


    train_loss /= len(train_data1)
    val_loss /= len(val_data1)

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

    if lang_pair1 == 'th-en':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_en_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair1 == 'th-vi':
        with open('hypothesis.txt', 'r') as input_file, open('bleu_score.txt', 'w') as bleu_file:
            check_call(["perl", "moses/generic/multi-bleu.perl", "-lc", th_vi_ref],
                       stdin=input_file, stdout=bleu_file)
    elif lang_pair1 == 'vi-en':
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

    torch.save(encoder.state_dict(), 'last_encoder_weight_multigoogle_%s_%s' % (lang_pair1, lang_pair2))
    torch.save(decoder.state_dict(), 'last_decoder_weight_multigoogle_%s_%s' % (lang_pair1, lang_pair2))

    # normally save for best loss but in this case not so indicative.
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     torch.save(encoder.state_dict(), 'best_loss_encoder_weight_%s-%s' % (lang_pair, mode))
    #     torch.save(decoder.state_dict(), 'best_loss_decoder_weight_%s-%s' % (lang_pair, mode))
    #     print('saving least val loss model from epoch [%d]' % (epoch + 1))

    if bleu_score > best_bleu_score:
        best_bleu_score = bleu_score
        torch.save(encoder.state_dict(), 'best_acc_encoder_weight_multigoogle_%s-%s-%s' % (lang_pair1, lang_pair2, mode))
        torch.save(decoder.state_dict(), 'best_acc_decoder_weight_multigoogle_%s-%s-%s' % (lang_pair1, lang_pair2, mode))
        print('saving most bleu acc  model from epoch [%d]' % (epoch + 1))