import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from utils import pad1d, mask_cnn, decoder_mask
import torch.nn.utils.rnn as rnn_utils

cudafloat = torch.cuda.FloatTensor
cudalong = torch.cuda.LongTensor

class CharEncoder(nn.Module):
    def __init__(self, embed_dim, N, dropout, k_num, k_size, poolstride,
                 en_bi, en_layers, en_H,
                 num_embed, highway_layers):
        super(CharEncoder, self).__init__()
        self.Ci = embed_dim
        self.k_num = k_num  # channel-out (number of filters)
        self.ks = zip(k_num, k_size)
        self.k_sum = sum(k_num)
        self.poolstride = poolstride
        self.bi = 2 if en_bi == True else 1
        self.en_H = en_H
        self.N = N
        self.en_layers = en_layers
        self.highway_layers = highway_layers

        self.embed = nn.Embedding(num_embed, embed_dim)

        self.convks = nn.ModuleList()
        for (num, size) in self.ks:
            # half convolution padding with two sided W-1 to get same input and output length
            self.convks.append(nn.Conv1d(in_channels=self.Ci,
                                         out_channels=num,
                                         kernel_size=size,
                                         stride=1))
        self.biGRU = nn.GRU(input_size=self.k_sum,
                            hidden_size=en_H,
                            num_layers=en_layers,
                            dropout=dropout,
                            bidirectional=en_bi)

        self.out_gate = nn.Linear(self.k_sum, self.k_sum)
        self.out_gate.bias.data.fill_(-2)
        self.t_gate = nn.Linear(self.k_sum, self.k_sum)
        self.t_gate.bias.data.fill_(-2)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax()

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.en_layers * self.bi, self.N, self.en_H).type(cudafloat))
        # c0 = Variable(torch.zeros(self.en_layers * self.bi, self.N, self.en_H).type(cudafloat))
        # return h0, c0
        return h0

    def pad_conv_and_pool(self, x, conv):
        # padding for half convolution (aka 'same' padding), which needs asymetric padding
        # asymmetric padding assumes front pad is longer. k_size=4 would have 2 zeros padded front and 1 zero padded back
        # pad1d takes (N,W,D)
        k_size = conv.kernel_size[0]
        if k_size > 1:
            total_pad = k_size - 1
            pad_front = math.ceil(total_pad / 2)
            pad_back = total_pad - pad_front
            x = pad1d(x, (pad_front, pad_back))
        x = F.relu(conv(x.transpose(1, 2)))  # (N,W,Ci) => (N,Co,W)
        result = F.max_pool1d(x, kernel_size=self.poolstride)  # (N, Co, W/s)

        return result

    def highway(self, x, num_layers):
        '''
        input - (W/s,N,D=k_sum)
        output -
        '''
        x.contiguous()
        x = x.view(-1, self.k_sum) #(W/s*N, D=k_sum)
        for i in range(num_layers):
            output = F.relu(self.out_gate(x))
            transform_gate = F.sigmoid(self.t_gate(x))
            carry_gate = 1 - transform_gate
            x = output*transform_gate + x*carry_gate
        return x

    def forward(self, input, hidden, x_lengths):
        # input - (N,W)
        x_mask_tensor = mask_cnn(x_lengths, self.poolstride)
        x_mask_tensor = x_mask_tensor.unsqueeze(1)

        x = self.embed(input)  # (N,W,D)
        x = [self.pad_conv_and_pool(x, convk)*x_mask_tensor for convk in self.convks]  # (N,Ci,W) => (N,Co,W/s)

        x = torch.cat(x, dim=1)  # (N, sum(Co) for all k_width, W/s)
        x = x.permute(2, 0, 1)  # (W/s,N,D=k_sum) prep for rnn
        x = self.highway(x, num_layers=self.highway_layers)

        output, hidden = self.biGRU(x.view(-1, self.N, self.k_sum), hidden)  # (W/s,N,H*bi)
        return output, hidden

class Encoder(nn.Module):
    def __init__(self, embed_dim, N, dropout, k_num, k_size, poolstride,
                 en_bi, en_layers, en_H,
                 num_embed):
        super(Encoder, self).__init__()
        self.bi = 2 if en_bi == True else 1
        self.en_H = en_H
        self.N = N
        self.embed_dim = embed_dim
        self.en_layers = en_layers
        self.num_embed = num_embed

        self.embed = nn.Embedding(self.num_embed, self.embed_dim)
        self.biGRU = nn.GRU(input_size=self.embed_dim,
                             hidden_size=en_H,
                             num_layers=en_layers,
                             dropout=dropout,
                             bidirectional=en_bi)

        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax()

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.en_layers * self.bi, self.N, self.en_H).type(cudafloat))
        # c0 = Variable(torch.zeros(self.en_layers * self.bi, self.N, self.en_H).type(cudafloat))
        # return h0, c0
        return h0

    def forward(self, input, hidden, x_lengths):
        # input - (N,W)
        x = self.embed(input.transpose(0, 1))  # (N,W) => (W,N,D)
        x = rnn_utils.pack_padded_sequence(x, x_lengths)
        output, hidden = self.biGRU(x, hidden)  # (W,N,H*bi)
        output, _ = rnn_utils.pad_packed_sequence(output)

        return output, hidden

class LuongDecoder(nn.Module):
    def __init__(self, de_H, en_Hbi, de_layers, dropout, de_bi, N, de_embed, out_sz):
        super(LuongDecoder, self).__init__()
        self.bi = 2 if de_bi == True else 1
        self.de_H = de_H
        self.en_Hbi = en_Hbi
        self.de_Hbi = self.bi * self.de_H
        self.de_layers = de_layers
        self.embed_sz = de_embed
        self.N = N
        self.out_sz = out_sz

        self.embedding = nn.Embedding(out_sz, de_embed)
        self.gru = nn.GRU(input_size=de_embed,
                          hidden_size=de_H,
                          num_layers=de_layers,
                          dropout=dropout,
                          bidirectional=de_bi)

        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # attention (Luong)
        # TODO: revise, won't work for all cases
        self.score_lin = nn.Linear(self.de_Hbi, self.de_Hbi)
        self.lin_comb = nn.Linear(self.de_Hbi + self.en_Hbi, self.de_Hbi)
        self.lin_out = nn.Linear(self.de_Hbi, self.out_sz)

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.de_layers * self.bi, self.N, self.de_H).type(cudafloat))
        # c0 = Variable(torch.zeros(self.de_layers * self.bi, self.N, self.de_H).type(cudafloat))
        # return h0, c0
        return h0

    def forward(self, inputs, encoder_out, hidden, x_mask_tensor):
        W_s, _, _ = encoder_out.size()

        # decoder RNN's output
        embed = self.embedding(inputs.transpose(0, 1))  # (N,W_t=1) => (W_t=1,N,D)
        W_t, _, _ = embed.size()

        # not sure how to make rnn_utils.pad_packed_sequence compatible with seq_len of 1, so roll out manual mask
        # y_mask_tensor = torch.cuda.FloatTensor(y_mask_list)  # (N)
        # y_mask_tensor = torch.unsqueeze(y_mask_tensor, 0) # (N) => (1,N)
        # y_mask_tensor = torch.unsqueeze(y_mask_tensor, 2) # (1,N) => (1,N,1)
        # y_mask_tensor = Variable(y_mask_tensor)


        y_mask_tensor = decoder_mask(inputs)
        y_mask_tensor = y_mask_tensor.unsqueeze(0)
        # print("inputs", inputs)
        # print("y mask tensor", y_mask_tensor)

        #TODO: Mask hidden state if target is pad
        rnn_output, hidden = self.gru(embed, hidden) #(W_t,N,H)
        hidden = hidden * y_mask_tensor  # (layers*bi, N, H) * (1,N,1).. using broadcasting


        rnn_output = rnn_output.transpose(0, 1)  # (N,W_t,H)
        rnn_output.contiguous()  # makes a contiguous copy for viewx

        # source hidden state
        # tensor containing the output features (h_s) from the last layer of the encoder RNN
        encoder_out = encoder_out.transpose(0, 1)  # (W_s,N,H) => (N,W_s,H)
        encoder_out.contiguous()  # (N,W_s,H)

        ### Luong's attn output & score

        # linear on RNN output
        h_t = self.score_lin(rnn_output.view(-1, self.de_Hbi))  # (N*W_t,H) dot (H,H)
        h_t = h_t.view(self.N, -1, self.de_Hbi)  # (N*W,H) => (N,W_t,H)
        h_s = encoder_out.permute(0, 2, 1)  # (N,W_s,H) => (N,H,W_s)

        # Matrix multiply between RNN output and Encoder's output
        scores = torch.bmm(h_t, h_s)  # (N,W_t,H) dot (N,H,W_s) => (N, W_t, W_s)

        # Normalize with softmax
        # TODO: Mask alignment (attn weights)
        align_vec = F.softmax(scores.view(-1, W_s), dim=1)  # softmax(N*W_t, Ws)
        align_vec = align_vec * x_mask_tensor
        align_vec = align_vec.view(self.N, -1, W_s)  # (N,W_t,W_s)


        # context_vec as weighted avg. of source states, based on attn weights
        context_vec = torch.bmm(align_vec, encoder_out) # (N,W_t,W_s) dot (N,W_s,H_en) => (N,W_t,H_en)
        concat_vec = torch.cat((context_vec, rnn_output), dim=2)  # (N,W_t,H_cat)
        concat_vec = concat_vec.view(-1, concat_vec.size()[2])  # (N,W_t,H_cat) => (N*W_t, H_cat)

        # linear, tanh
        attn_h = self.lin_comb(concat_vec)  # (N*W_t, H*2) => (N*W_t,H)
        attn_h = attn_h.view(self.N, W_t, self.de_Hbi)  # (N,W_t,H)
        attn_h = F.tanh(attn_h.transpose(0, 1))  # (W_t,N,H)

        # Linear, softmax
        output = attn_h
        output = self.lin_out(output.view(-1, self.de_Hbi))  # (W_t,N,H) => (W_t*N,H)
        output = self.logsoftmax(output)  # (W*N,H) => (W_t*N,Out)
        output = output.view(W_t, self.N, self.out_sz)  # (W_t*N,Out) => (W_t,N,Out)

        #TODO: Mask decoder output
        # print("output before mask", output)
        output = output * y_mask_tensor
        # print("output after mask", output)

        return output, hidden, align_vec.transpose(0, 1)

