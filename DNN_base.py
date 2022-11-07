# -*- coding: utf-8 -*-
"""
Created on 2021.06.16
modified on 2021.11.11
@author: LXA
"""
import torch
import torch.nn as tn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt

"""
通常来说 torch.nn.functional 调用了 THNN 库，实现核心计算，但是不对 learnable_parameters 例如 weight bias ，进行管理，
为模型的使用带来不便。而 torch.nn 中实现的模型则对 torch.nn.functional，本质上是官方给出的对 torch.nn.functional的使用范例，
我们通过直接调用这些范例能够快速方便的使用 pytorch ，但是范例可能不能够照顾到所有人的使用需求，因此保留 torch.nn.functional 
来为这些用户提供灵活性，他们可以自己组装需要的模型。因此 pytorch 能够在灵活性与易用性上取得平衡。

特别注意的是，torch.nn不全都是对torch.nn.functional的范例，有一些调用了来自其他库的函数，例如常用的RNN型神经网络族即没有
在torch.nn.functional中出现。
参考链接：
        https://blog.csdn.net/gao2628688/article/details/99724617
"""


class my_actFunc(tn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def forward(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tnf.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tnf.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':   # 增强的Tanh函数 Enhance Tanh
            out_x = torch.tanh(0.5*torch.pi*x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*torch.sin(x_input) + 0.5*torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        elif str.lower(self.actName) == 'gelu':
            out_x = tnf.gelu(x_input)
        else:
            out_x = x_input
        return out_x


# ----------------dense net(constructing NN and initializing weights and bias )------------
class DenseNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(DenseNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        if str.lower(self.name2Model) == 'fourier_dnn':
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=False)
            tn.init.xavier_normal_(input_layer.weight)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                if i_layer == 0:
                    hidden_layer = tn.Linear(in_features=2 * hidden_units[i_layer],
                                             out_features=hidden_units[i_layer+1])
                    tn.init.xavier_normal_(hidden_layer.weight)
                    tn.init.uniform_(hidden_layer.bias, -1, 1)
                else:
                    hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                    tn.init.xavier_normal_(hidden_layer.weight)
                    tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, 0, 1)
            self.dense_layers.append(out_layer)
        else:
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, -1, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, -1, 1)
            self.dense_layers.append(out_layer)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        self.use_gpu = to_gpu
        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.square(layer.weight))
                regular_b = regular_b + torch.sum(torch.square(layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def get_regular_sum2Fourier(self, regular_model='L2'):
        regular_w = 0.0
        regular_b = 0.0
        i_layer = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.abs(layer.bias))
                i_layer = i_layer + 1
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
                i_layer = i_layer + 1
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=0.5):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        # print(dense_in)
        H = dense_in(inputs)
        if str.lower(self.name2Model) == 'dnn':
            H = self.actFunc_in(H)
        else:
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            torch_mixcoe = torch.from_numpy(mixcoe)
            if self.to_gpu:
                torch_mixcoe = torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

            if str.lower(self.name2Model) == 'fourier_dnn':
                assert(self.actFunc_in.actName == 'fourier')
                H = sFourier*self.actFunc_in(H*torch_mixcoe)
            else:
                H = self.actFunc_in(H * torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            # print(dense_layer)
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        # print(dense_out)
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Dense_Net(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        if str.lower(self.name2Model) == 'fourier_dnn':
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], )
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, 0, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                if i_layer == 0:
                    hidden_layer = tn.Linear(in_features=2.0 * hidden_units[i_layer],
                                             out_features=hidden_units[i_layer+1], bias=False)
                    tn.init.xavier_normal_(hidden_layer.weight)
                else:
                    hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                    tn.init.xavier_normal_(hidden_layer.weight)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, 0, 1)
            self.dense_layers.append(out_layer)
        else:
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, -1, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, -1, 1)
            self.dense_layers.append(out_layer)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        self.use_gpu = to_gpu
        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.square(layer.weight))
                regular_b = regular_b + torch.sum(torch.square(layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        # print(dense_in)
        H = dense_in(inputs)
        if str.lower(self.name2Model) == 'fourier_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = torch.cat([torch.cos(H*mixcoe), torch.sin(H*mixcoe)], dim=-1)
        elif str.lower(self.name2Model) == 'scale_dnn' or str.lower(self.name2Model) == 'wavelet_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = self.actFunc_in(H*mixcoe)
        else:
            H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            # print(dense_layer)
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        # print(dense_out)
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Pure_DenseNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', to_gpu=False, gpu_no=0):
        super(Pure_DenseNet, self).__init__()
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        self.use_gpu = to_gpu
        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)
        H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------Dense_ScaleNet(constructing NN and initializing weights and bias )------------
class Dense_ScaleNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0])
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        self.use_gpu = to_gpu
        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))
        H = self.actFunc_in(H*torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Dense_FourierNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', to_gpu=False, gpu_no=0,
                 repeat_Highfreq=True):
        super(Dense_FourierNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=False)
        tn.init.xavier_normal_(input_layer.weight)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            if i_layer == 0:
                hidden_layer = tn.Linear(in_features=2 * hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            else:
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1])
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        self.use_gpu = to_gpu
        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0.0
        regular_b = 0.0
        i_layer = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.abs(layer.bias))
                i_layer = i_layer + 1
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
                i_layer = i_layer + 1
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))

        H = sFourier*torch.cat([torch.cos(H*torch_mixcoe), torch.sin(H*torch_mixcoe)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


def func_test(x, in_dim=2, equa='eq1'):
    if in_dim == 1 and equa == 'eq1':
        out = np.sin(np.pi * x[:, 0]) + 0.1 * np.sin(3 * np.pi * x[:, 0]) + 0.01 * np.sin(10 * np.pi * x[:, 0])
    if in_dim == 1 and equa == 'eq2':
        out = np.sin(np.pi * x[:, 0]) + 0.1 * np.sin(3 * np.pi * x[:, 0]) + \
              0.05 * np.sin(10 * np.pi * x[:, 0]) + 0.01 * np.sin(50 * np.pi * x[:, 0])
    elif in_dim == 2:
        out = np.sin(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) + 0.1 * np.sin(5 * np.pi * x[:, 0] * x[:, 0] + 3 * np.pi * x[:, 1] * x[:, 1]) + \
              0.01 * np.sin(15 * np.pi * x[:, 0] * x[:, 0] + 20 * np.pi * x[:, 1] * x[:, 1])

    out = np.reshape(out, newshape=(-1, 1))
    return out


# This is an example for using the above module
class DNN2TEST(tn.Module):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='tanh',
                 name2actHidden='tanh', name2actOut='linear', factor2freq=None, sFourier=1.0, use_gpu=False, no2gpu=0,
                 opt2regular_WB='L0', type2numeric='float32', repeat_Highfreq=True):
        super(DNN2TEST, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB
        if str.upper(Model_name) == 'DNN':
            self.DNN = Pure_DenseNet(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                     actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                     type2float=type2numeric, to_gpu=use_gpu, gpu_no=no2gpu)
        elif str.upper(Model_name) == 'SCALE_DNN':
            self.DNN = Dense_ScaleNet(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                      actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                      type2float=type2numeric, to_gpu=use_gpu, gpu_no=no2gpu,
                                      repeat_Highfreq=repeat_Highfreq)
        elif str.upper(Model_name) == 'FOURIER_DNN':
            self.DNN = Dense_FourierNet(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                                        actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                        type2float=type2numeric, to_gpu=use_gpu, gpu_no=no2gpu,
                                        repeat_Highfreq=repeat_Highfreq)

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(no2gpu)
        else:
            self.opt2device = 'cpu'

    def forward(self, x_input):
        out = self.DNN(x_input, scale=self.factor2freq)
        return out

    def get_sum2wB(self):
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def cal_l2loss(self, x_input=None, y_input=None):
        out = self.DNN(x_input, scale=self.factor2freq)
        squre_loss = torch.mul(y_input - out, y_input - out)
        loss = torch.mean(squre_loss, dim=0)
        return loss, out


def test_DNN():
    batch_size = 100
    # dim_in = 1
    dim_in = 2
    dim_out = 1
    act_func2In = 'tanh'

    # act_func2Hidden = 'tanh'
    # act_func2Hidden = 'Enhance_tanh'
    act_func2Hidden = 'sin'
    # act_func2Hidden = 'sinAddcos'
    # act_func2Hidden = 'gelu'

    act_func2Out = 'linear'

    hidden_list = (50, 30, 10, 10, 20)
    freq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float32)
    # model_name = 'DNN'
    model_name = 'Fourier_DNN'
    init_lr = 0.01
    max_it = 100000
    with_gpu = True
    highFreq_repeat = True

    model = DNN2TEST(input_dim=dim_in, out_dim=dim_out, hidden_layer=hidden_list, Model_name=model_name,
                     name2actIn=act_func2In, name2actHidden=act_func2Hidden, name2actOut=act_func2Out, factor2freq=freq,
                     sFourier=1.0, use_gpu=with_gpu, no2gpu=0, repeat_Highfreq=highFreq_repeat)
    if with_gpu:
        model = model.cuda(device='cuda:'+str(0))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.995)
    arr2epoch = []
    arr2loss = []
    arr2lr = []
    for i_epoch in range(max_it):
        x = np.random.rand(batch_size, dim_in)
        x = x.astype(dtype=np.float32)
        torch_x = torch.from_numpy(x)
        y = func_test(x, in_dim=dim_in, equa='eq2')
        torch_y = torch.from_numpy(y)
        if with_gpu:
            torch_x = torch_x.cuda(device='cuda:'+str(0))
            torch_y = torch_y.cuda(device='cuda:' + str(0))

        loss2data, prediction = model.cal_l2loss(x_input=torch_x, y_input=torch_y)
        sum2wb = model.get_sum2wB()
        loss = loss2data + sum2wb

        optimizer.zero_grad()      # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()            # 求偏导
        optimizer.step()           # 更新参数
        scheduler.step()

        if i_epoch % 100 == 0:
            print('i_epoch ------- loss:', i_epoch, loss.item())
            print("第%d个epoch的学习率：%.10f" % (i_epoch, optimizer.param_groups[0]['lr']))
            arr2epoch.append(int(i_epoch/100))
            arr2loss.append(loss.item())
            arr2lr.append(optimizer.param_groups[0]['lr'])

    if model_name == 'DNN':
        plt.figure()
        ax = plt.gca()
        plt.plot(arr2loss, 'b-.', label='loss')
        plt.xlabel('epoch/100', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(fontsize=18)
        plt.title('DNN')
        ax.set_yscale('log')
        plt.show()
    else:
        plt.figure()
        ax = plt.gca()
        plt.plot(arr2loss, 'b-.', label='loss')
        plt.xlabel('epoch/100', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(fontsize=18)
        plt.title('Fourier_DNN')
        ax.set_yscale('log')
        plt.show()

    # plt.cla()
    # plt.plot(x[:, 0], x[:, 1], y, 'b*')
    # plt.show()


if __name__ == "__main__":
    test_DNN()

