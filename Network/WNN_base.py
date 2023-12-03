# -*- coding: utf-8 -*-
"""
Created on 2022.11.11
@author: LXA
"""
import torch
import torch.nn as tn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt


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


# ----------------------------------网络模型: DNN model with radius basis function--------------------------------------
class DNN_RBFBase(tn.Module):
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
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', repeat_Highfreq=True,
                 to_gpu=False, gpu_no=0, opt2init_W2RBF='uniform_random', value_min2weight=0.0, value_max2weight=0.75,
                 shuffle_W2RBF=True, opt2init_C2RBF = 'uniform_random', shuffle_B2RBF=True, value_min2bias=-1.0,
                 value_max2bias=1.0):
        super(DNN_RBFBase, self).__init__()
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

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

        self.dense_layers = tn.ModuleList()
        self.Center2RBF = tn.Parameter()

        # 为了取得好的结果，第一层 W和B都要使用 uniform_random 进行初始化，使用 gauss_random 初始化的效果不好
        # 第一层的 Weight 其实是 RBF 的 B， 第一层的 Bias 的转置 是 RBF 的 W
        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=True,
                                dtype=self.float_type, device=self.opt2device)
        self.dense_layers.append(input_layer)

        if str.lower(opt2init_W2RBF) == 'uniform_random':
            tn.init.uniform_(input_layer.bias, value_min2weight, value_max2weight)
        elif str.lower(opt2init_W2RBF) == 'normal_random':
            tn.init.normal_(input_layer.bias, 0.5*(value_min2weight+value_max2weight), 1.0)
        else:
            tn.init.xavier_normal_(input_layer.bias)    # 使用Xavier初始化的时候，被初始化的对象需要是矩阵(两个维度)

        if str.lower(opt2init_C2RBF) == 'uniform_random':
            tn.init.uniform_(input_layer.weight, value_min2bias, value_max2bias)
        elif str.lower(opt2init_C2RBF) == 'normal_random':
            tn.init.normal_(input_layer.weight, 0.5*(value_min2bias+value_max2bias), 1.0)
        else:
            tn.init.xavier_normal_(input_layer.weight)

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

        if 2 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 3 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 4 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec4 = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 5 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec4 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec5 = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec4 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec5 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)

    def get_regular_sum2WB(self, regular_model='L2'):
        regular_w = 0.0
        regular_b = 0.0
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

    def forward(self, inputs, scale=None, sWavebase=1.0, training=None, mask=None):
        # ------------------- dealing with the input data -----------------------
        dense_in = self.dense_layers[0]

        # 频率标记按按照比例复制
        # np.repeat(a, repeats, axis=None)
        # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
        # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
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

        W2RBF = torch.reshape(dense_in.bias, shape=[1, -1])
        B2RBF = torch.transpose(dense_in.weight, 1, 0)

        if 1 == self.indim:
            diff2X = inputs - B2RBF
            norm2diffx = torch.square(diff2X)
        elif 2 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            norm2diffx = square_diff_X1 + square_diff_X2
        elif 3 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3
        elif 4 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)
            X4 = torch.matmul(inputs, self.X_vec4)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)
            B4 = torch.matmul(self.B_vec4, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            diff_X4 = X4 - B4
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            square_diff_X4 = torch.square(diff_X4)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3 + square_diff_X4   # (?, N)
        elif 5 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)
            X4 = torch.matmul(inputs, self.X_vec4)
            X5 = torch.matmul(inputs, self.X_vec5)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)
            B4 = torch.matmul(self.B_vec4, B2RBF)
            B5 = torch.matmul(self.B_vec5, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            diff_X4 = X4 - B4
            diff_X5 = X5 - B5
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            square_diff_X4 = torch.square(diff_X4)
            square_diff_X5 = torch.square(diff_X5)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3 + square_diff_X4 + square_diff_X5   # (?, N)
        weight_norm2diff_X = torch.mul(norm2diffx, W2RBF)

        if len(scale) == 1:
            # H = torch.sin(weight_norm2diff_X)
            H = torch.exp(-0.5 * weight_norm2diff_X)
            # H = torch.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.cos(1.75 * norm2diffx)
            # H = torch.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.cos(1.75 * weight_norm2diff_X)
            # H = torch.exp(-0.5 *weight_norm2diff_X) * sRBF*(tf.cos(1.75 * weight_norm2diff_X) + tf.sin(1.75 * weight_norm2diff_X))
        else:
            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.sin(norm2diffx * mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(norm2diffx)

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(norm2diffx * mixcoe)     # 这个是最好的函数，对于2维效果好，对于1维不好
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * (tf.sin(norm2diffx) + tf.cos(norm2diffx))
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X * mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(np.pi * norm2diffx * mixcoe)  # work 不好
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * (tf.cos(norm2diffx * mixcoe) + tf.sin(norm2diffx * mixcoe))
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5*(tf.cos(1.75 * norm2diffx * mixcoe) + tf.sin(1.75 * norm2diffx * mixcoe))

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * tf.cos(1.75 * norm2diffx)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * tf.cos(1.75 * norm2diffx * torch_mixcoe)   # 这个是次好的函数
            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * torch.cos(1.75 * weight_norm2diff_X)

            H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * \
                sWavebase * torch.cos(1.75 * weight_norm2diff_X * torch_mixcoe)  # 这个是最好的函数，对于1和2维效果都好

            # H = tf.exp(-0.5 *weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * \
            #     (tf.cos(weight_norm2diff_X * torch_mixcoe) + tf.sin(weight_norm2diff_X * torch_mixcoe))

            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * 0.5*(torch.cos(weight_norm2diff_X) + torch.sin(weight_norm2diff_X))

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) + \
            #     sWavebase * 0.5 * (tf.cos(weight_norm2diff_X * mixcoe) + tf.sin(weight_norm2diff_X * mixcoe))        # 不好

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer + 1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------------------------网络模型: DNN model with radius basis function--------------------------------------
class RBFNN_Base(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', repeat_Highfreq=True,
                 to_gpu=False, gpu_no=0, opt2init_W2RBF='uniform_random', value_min2weight=0.0, value_max2weight=0.75,
                 shuffle_W2RBF=True, opt2init_C2RBF = 'uniform_random', shuffle_B2RBF=True, value_min2bias=-1.0,
                 value_max2bias=1.0):
        super(RBFNN_Base, self).__init__()
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = tn.ModuleList()

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

        # 为了取得好的结果，weight 和 center 都要使用 uniform_random 进行初始化，使用 gauss_random 初始化的效果不好
        centroid_temp2tensor = torch.empty((indim, hidden_units[0]), dtype=self.float_type, device=self.opt2device)
        self.centroid2RBF = tn.Parameter(centroid_temp2tensor, requires_grad=True)

        if str.lower(opt2init_C2RBF) == 'uniform_random':
            tn.init.uniform_(self.centroid2RBF, value_min2bias, value_max2bias)
        elif str.lower(opt2init_C2RBF) == 'normal_random':
            tn.init.normal_(self.centroid2RBF, 0.5*(value_min2bias+value_max2bias), 1.0)
        else:
            stddev2center = (2.0 / (indim + hidden_units[0])) ** 0.5
            tn.init.normal_(self.center2RBF, mean=0.0, std=stddev2center)

        weight_temp2tensor = torch.empty((1, hidden_units[0]), dtype=self.float_type, device=self.opt2device)
        self.weight2RBF = tn.Parameter(weight_temp2tensor, requires_grad=False)
        tn.init.zeros_(self.weight2RBF)

        if str.lower(opt2init_W2RBF) == 'uniform_random':
            tn.init.uniform_(self.weight2RBF, value_min2weight, value_max2weight)
        elif str.lower(opt2init_W2RBF) == 'normal_random':
            tn.init.normal_(self.weight2RBF, 0.5 * (value_min2weight + value_max2weight), 1.0)
        else:
            stddev2weight = (2.0 / (1 + hidden_units[0])) ** 0.5
            tn.init.normal_(self.weight2RBF, mean=0.0, std=stddev2weight)

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

        if 2 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 3 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 4 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec4 = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec4 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)
        elif 5 == indim:
            self.X_vec1 = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec2 = torch.tensor([[0.0], [1.0], [0.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec3 = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec4 = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0]], dtype=self.float_type, device=self.opt2device)
            self.X_vec5 = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0]], dtype=self.float_type, device=self.opt2device)

            self.B_vec1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec2 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec3 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec4 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]], dtype=self.float_type, device=self.opt2device)
            self.B_vec5 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=self.float_type, device=self.opt2device)

    def get_regular_sum2WB(self, regular_model='L2'):
        regular_w = 0.0
        regular_b = 0.0
        if regular_model == 'L1':
            regular_w = torch.sum(torch.abs(self.weight2RBF))
            regular_b = torch.sum(torch.abs(self.centroid2RBF))
        elif regular_model == 'L2':
            regular_w = torch.sum(torch.square(self.weight2RBF))
            regular_b = torch.sum(torch.square(self.center2RBF))

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

    def forward(self, inputs, scale=None, sWavebase=1.0, training=None, mask=None):
        # 频率标记按按照比例复制
        # np.repeat(a, repeats, axis=None)
        # 输入: a是数组,repeats是各个元素重复的次数(repeats一般是个标量,稍复杂点是个list),在axis的方向上进行重复
        # 返回: 如果不指定axis,则将重复后的结果展平(维度为1)后返回;如果指定axis,则不展平
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

        W2RBF = self.weight2RBF
        B2RBF = self.centroid2RBF

        # ------------------- dealing with the input data ----------------------
        if 1 == self.indim:
            diff2X = inputs - B2RBF
            norm2diffx = torch.square(diff2X)
        elif 2 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            norm2diffx = square_diff_X1 + square_diff_X2
        elif 3 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3
        elif 4 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)
            X4 = torch.matmul(inputs, self.X_vec4)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)
            B4 = torch.matmul(self.B_vec4, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            diff_X4 = X4 - B4
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            square_diff_X4 = torch.square(diff_X4)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3 + square_diff_X4   # (?, N)
        elif 5 == self.indim:
            X1 = torch.matmul(inputs, self.X_vec1)
            X2 = torch.matmul(inputs, self.X_vec2)
            X3 = torch.matmul(inputs, self.X_vec3)
            X4 = torch.matmul(inputs, self.X_vec4)
            X5 = torch.matmul(inputs, self.X_vec5)

            B1 = torch.matmul(self.B_vec1, B2RBF)
            B2 = torch.matmul(self.B_vec2, B2RBF)
            B3 = torch.matmul(self.B_vec3, B2RBF)
            B4 = torch.matmul(self.B_vec4, B2RBF)
            B5 = torch.matmul(self.B_vec5, B2RBF)

            diff_X1 = X1 - B1
            diff_X2 = X2 - B2
            diff_X3 = X3 - B3
            diff_X4 = X4 - B4
            diff_X5 = X5 - B5
            square_diff_X1 = torch.square(diff_X1)
            square_diff_X2 = torch.square(diff_X2)
            square_diff_X3 = torch.square(diff_X3)
            square_diff_X4 = torch.square(diff_X4)
            square_diff_X5 = torch.square(diff_X5)
            norm2diffx = square_diff_X1 + square_diff_X2 + square_diff_X3 + square_diff_X4 + square_diff_X5   # (?, N)
        weight_norm2diff_X = torch.mul(norm2diffx, W2RBF)

        if len(scale) == 1:
            # H = torch.sin(weight_norm2diff_X)
            H = torch.exp(-0.5 * weight_norm2diff_X)
            # H = torch.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.cos(1.75 * norm2diffx)
            # H = torch.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.cos(1.75 * weight_norm2diff_X)
            # H = torch.exp(-0.5 *weight_norm2diff_X) * sRBF*(tf.cos(1.75 * weight_norm2diff_X) + tf.sin(1.75 * weight_norm2diff_X))
        else:
            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X) * sRBF * tf.sin(norm2diffx * mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(norm2diffx)

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(norm2diffx * mixcoe)     # 这个是最好的函数，对于2维效果好，对于1维不好
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * (tf.sin(norm2diffx) + tf.cos(norm2diffx))
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(weight_norm2diff_X * mixcoe)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * tf.sin(np.pi * norm2diffx * mixcoe)  # work 不好
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * (tf.cos(norm2diffx * mixcoe) + tf.sin(norm2diffx * mixcoe))
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5*(tf.cos(1.75 * norm2diffx * mixcoe) + tf.sin(1.75 * norm2diffx * mixcoe))

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * tf.cos(1.75 * norm2diffx)
            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * tf.cos(1.75 * norm2diffx * torch_mixcoe)   # 这个是次好的函数
            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * torch.cos(1.75 * weight_norm2diff_X)

            H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * \
                sWavebase * torch.cos(1.75 * weight_norm2diff_X * torch_mixcoe)  # 这个是最好的函数，对于1和2维效果都好

            # H = tf.exp(-0.5 *weight_norm2diff_X * torch_mixcoe) * sRBF * 0.5 * \
            #     (tf.cos(weight_norm2diff_X * torch_mixcoe) + tf.sin(weight_norm2diff_X * torch_mixcoe))

            # H = torch.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) * sWavebase * 0.5*(torch.cos(weight_norm2diff_X) + torch.sin(weight_norm2diff_X))

            # H = tf.exp(-0.5 * weight_norm2diff_X * torch_mixcoe) + \
            #     sWavebase * 0.5 * (tf.cos(weight_norm2diff_X * mixcoe) + tf.sin(weight_norm2diff_X * mixcoe))        # 不好

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# This is an example for using the above wave_base network module
class WaveDNN(tn.Module):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='tanh',
                 name2actHidden='tanh', name2actOut='linear', type2numeric='float32', repeat_highFreq=True,
                 use_gpu=False, no2gpu=0, factor2freq=None, sWavebase=1.0,  opt2regular_WB='L0', min_value2weight=0.0,
                 max_value2weight=1.0, min_value2Centroid=-1.0, max_value2Centroid=1.0, opt2init_W='uniform_random',
                 opt2init_Centroid='uniform_random'):
        super(WaveDNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.type2numeric = type2numeric
        self.use_gpu = use_gpu
        self.no2gpu = no2gpu
        self.factor2freq = factor2freq
        self.sWavebase = sWavebase
        self.opt2regular_WB = opt2regular_WB

        self.DNN = RBFNN_Base(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                               actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                               type2float=type2numeric, repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=no2gpu,
                               opt2init_W2RBF=opt2init_W, value_min2weight=min_value2weight,
                               value_max2weight=max_value2weight, shuffle_W2RBF=True,
                               opt2init_C2RBF=opt2init_Centroid, shuffle_B2RBF=True,
                               value_min2bias=min_value2Centroid, value_max2bias=max_value2Centroid)

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
        assert (x_input is not None)
        out = self.DNN(x_input, scale=self.factor2freq)
        return out

    def get_sum2wB(self):
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def cal_l2loss(self, x_input=None, y_input=None):
        assert (x_input is not None)
        assert (y_input is not None)
        out = self.DNN(x_input, scale=self.factor2freq, sWavebase=self.sWavebase)
        squre_loss = torch.mul(y_input - out, y_input - out)
        loss = torch.mean(squre_loss, dim=0)
        return loss, out

    def evalue_WaveDNN(self, x_input=None):
        assert(x_input is not None)
        out = self.DNN(x_input, scale=self.factor2freq, sWavebase=self.sWavebase)
        return out


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


def test_WaveNN():
    batch_size = 100
    # dim_in = 1
    dim_in = 2
    dim_out = 1
    act_func2In = 'tanh'

    # act_func2Hidden = 'tanh'
    # act_func2Hidden = 'Enhance_tanh'
    # act_func2Hidden = 'sin'
    # act_func2Hidden = 'sinAddcos'
    act_func2Hidden = 'gelu'

    act_func2Out = 'linear'

    hidden_list = (200, 20, 10, 10, 5)
    freq = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float32)   # 无论Fourier 还是 WaveLetNN, 因子越大，表示频率越高。因子越小，表示频率月底
    model_name = 'WaveDNN'
    init_lr = 0.01
    # max_it = 10000
    max_it = 100000
    with_gpu = True
    highFreq_repeat = True

    model = WaveDNN(input_dim=dim_in, out_dim=dim_out, hidden_layer=hidden_list, Model_name=model_name,
                    name2actIn=act_func2In, name2actHidden=act_func2Hidden, name2actOut=act_func2Out,
                    repeat_highFreq=highFreq_repeat, use_gpu=with_gpu, no2gpu=0, factor2freq=freq,  opt2regular_WB='L0',
                    sWavebase=1.0,  min_value2weight=0.0, max_value2weight=1.0,
                    min_value2Centroid=-1.0, max_value2Centroid=1.0, opt2init_W='uniform_random',
                    opt2init_Centroid='uniform_random')
    if with_gpu:
        model = model.cuda(device='cuda:'+str(0))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.01)
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
        regular_sum2wb = model.get_sum2wB()
        loss = loss2data + regular_sum2wb

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 求偏导
        optimizer.step()       # 更新参数
        scheduler.step()

        if i_epoch % 100 == 0:
            print('i_epoch ------- loss:', i_epoch, loss.item())
            print("第%d个epoch的学习率：%.10f" % (i_epoch, optimizer.param_groups[0]['lr']))
            arr2epoch.append(int(i_epoch/100))
            arr2loss.append(loss.item())
            arr2lr.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    ax = plt.gca()
    plt.plot(arr2loss, 'b-.', label='loss')
    plt.xlabel('epoch/100', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=18)
    plt.title('WaveDNN')
    ax.set_yscale('log')
    plt.show()

    # plt.cla()
    # plt.plot(x[:, 0], x[:, 1], y, 'b*')
    # plt.show()


if __name__ == "__main__":
    test_WaveNN()