# -*- coding: utf-8 -*-
"""
Created on 2021.06.18
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
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input) * tnf.relu(1 - x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        else:
            out_x = x_input
        return out_x


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

    def get_regular_sum2WB(self, regular_model='L2'):
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


class DNN_test(tn.Module):
    def __init__(self, dim_in=2, dim_out=1, hidden_layers=None, name2Model='DNN', actName_in='tanh',
                 actName_hidden='tanh', actName_out='linear', use_gpu=False, no2gpu=0):
        super(DNN_test, self).__init__()
        self.name2Model = name2Model
        self.dim_in = dim_in
        self.dim_out = dim_out
        if name2Model == 'DNN':
            self.DNN = Pure_DenseNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_layers, name2Model=name2Model,
                                     actName2in=actName_in, actName=actName_hidden, actName2out=actName_out)

    def forward(self, x_input, freq=None):
        out = self.DNN(x_input, scale=freq)
        return out

    def get_sum2wB(self):
        sum2WB = self.DNN.get_regular_sum2WB()
        return sum2WB

    def cal_l2loss(self, x_input=None, freq=None, y_input=None):
        out = self.DNN(x_input, scale=freq)
        squre_loss = torch.mul(y_input - out, y_input - out)
        loss = torch.mean(squre_loss, dim=0)
        return loss, out


def test_DNN():
    batch_size = 10
    dim_in = 2
    dim_out = 1
    hidden_list = (10, 20, 10, 10, 20)
    freq = np.array([1, 2, 3, 4], dtype=np.float32)
    model_name = 'DNN'
    init_lr = 0.01
    max_it = 10000
    with_gpu = False

    model = DNN_test(dim_in=dim_in, dim_out=dim_out, hidden_layers=hidden_list, name2Model=model_name,
                     actName_in='tanh', actName_hidden='tanh', use_gpu=with_gpu, no2gpu=0)
    if with_gpu:
        model = model.cuda(device='cuda:' + str(0))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    # optimizer = torch.optim.Adam(params2Net, lr=init_lr)  # Adam
    optimizer = torch.optim.LBFGS(params2Net, lr=init_lr, max_iter=5)         # LBFGS

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    arr2epoch = []
    arr2loss = []
    arr2lr = []
    for i_epoch in range(max_it):
        x = np.random.rand(batch_size, dim_in)
        x = x.astype(dtype=np.float32)
        torch_x = torch.from_numpy(x)
        y = np.reshape(np.sin(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]), newshape=(-1, 1))
        torch_y = torch.from_numpy(y)
        if with_gpu:
            torch_x = torch_x.cuda(device='cuda:' + str(0))
            torch_y = torch_y.cuda(device='cuda:' + str(0))

        loss_temp, prediction = model.cal_l2loss(x_input=torch_x, freq=freq, y_input=torch_y)
        sum2wb = model.get_sum2wB()
        loss = loss_temp + sum2wb

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()        # 求偏导
        optimizer.step()       # 更新参数
        scheduler.step()       # 更新学习率

        if i_epoch % 100 == 0:
            print('i_epoch --- loss:', i_epoch, loss.item())
            # print("第%d个epoch的学习率：%f" % (i_epoch, optimizer.param_groups[0]['lr']))
            arr2loss.append(loss.item())
            arr2lr.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    ax = plt.gca()
    plt.plot(arr2loss, 'b-.', label='loss')
    plt.xlabel('epoch/100', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=18)
    ax.set_yscale('log')
    plt.show()

    # plt.cla()
    # plt.plot(x[:, 0], x[:, 1], y, 'b*')
    # plt.show()


def test_DNN2():
    batch_size = 10
    dim_in = 2
    dim_out = 1
    hidden_list = (10, 20, 10, 10, 20)
    freq = np.array([1, 2, 3, 4], dtype=np.float32)
    model_name = 'DNN'
    init_lr = 0.01
    max_it = 10000
    with_gpu = False

    model = DNN_test(dim_in=dim_in, dim_out=dim_out, hidden_layers=hidden_list, name2Model=model_name,
                     actName_in='tanh', actName_hidden='tanh', use_gpu=with_gpu, no2gpu=0)
    if with_gpu:
        model = model.cuda(device='cuda:' + str(0))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    # optimizer = torch.optim.Adam(params2Net, lr=init_lr)  # Adam
    optimizer = torch.optim.LBFGS(params2Net, lr=init_lr, max_iter=5)         # LBFGS

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    arr2epoch = []
    arr2loss = []
    arr2lr = []
    for i_epoch in range(max_it):
        x = np.random.rand(batch_size, dim_in)
        x = x.astype(dtype=np.float32)
        torch_x = torch.from_numpy(x)
        y = np.reshape(np.sin(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]), newshape=(-1, 1))
        torch_y = torch.from_numpy(y)
        if with_gpu:
            torch_x = torch_x.cuda(device='cuda:' + str(0))
            torch_y = torch_y.cuda(device='cuda:' + str(0))

        def closure():
            optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
            loss_temp, prediction = model.cal_l2loss(x_input=torch_x, freq=freq, y_input=torch_y)
            sum2wb = model.get_sum2wB()
            loss = loss_temp + sum2wb
            loss.backward()        # 求偏导
            return loss

        optimizer.step(closure)    # 更新参数
        scheduler.step()           # 更新学习率

        # if i_epoch % 100 == 0:
        #     print('i_epoch --- loss:', i_epoch, loss.item())
        #     # print("第%d个epoch的学习率：%f" % (i_epoch, optimizer.param_groups[0]['lr']))
        #     arr2loss.append(loss.item())
        #     arr2lr.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    ax = plt.gca()
    plt.plot(arr2loss, 'b-.', label='loss')
    plt.xlabel('epoch/100', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend(fontsize=18)
    ax.set_yscale('log')
    plt.show()

    # plt.cla()
    # plt.plot(x[:, 0], x[:, 1], y, 'b*')
    # plt.show()


if __name__ == "__main__":
    # test_DNN()
    test_DNN2()

