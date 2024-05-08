"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2024 年 4月 12 日
 Final version: 2024年 5 月 8 日
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil

import time
import datetime
from Network import DNN_base

from Problems import General_Laplace

from utilizers import DNN_Log_Print
from utilizers import dataUtilizer2torch
from utilizers import DNN_tools
from utilizers import plotData
from utilizers import saveData
from utilizers import Load_data2Mat
from utilizers import save_load_NetModule


class PDE_DNN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 scales=None, sFourier=1.0, repeat_highFreq=True, use_gpu=False, No2GPU=0):
        """
        initialing the class of MscaleDNN with given setups
        Args:
             input_dim:        the dimension of input data
             out_dim:          the dimension of output data
             hidden_layer:     the number of units for hidden layers(a tuple or a list)
             Model_name:       the name of DNN model(DNN, ScaleDNN or FourierDNN)
             name2actIn:       the name of activation function for input layer
             name2actHidden:   the name of activation function for all hidden layers
             name2actOut:      the name of activation function for all output layer
             opt2regular_WB:   the option of regularing weights and biases
             type2numeric:     the numerical type of float
             scales:           the scale vector for ScaleDNN or FourierDNN
             sFourier:         the relaxation factor for FourierDNN
             repeat_highFreq:  repeat the high-frequency scale-factor or not
             use_gpu:          using cuda or not
             No2GPU:           if your computer have more than one GPU, please assign the number of GPU
        """
        super(PDE_DNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_SUB_DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU, num2subnets=len(scales))

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.scales = scales
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'

    def loss2in(self, XYZ=None, fside=None, if_lambda2fside=True, loss_type='ritz_loss', relaxation2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) in the interior points for given domain
        -Laplace U = f,       in Omega
        BU = g                on Partial Omega, where B is a boundary operator
        Args:
             XYZ:             the input data of variable. ------  float, shape=[B,D]
             fside:           the force side              ------  float, shape=[B,1]
             if_lambda2fside: the force-side is a lambda function or an array  ------ Bool
             loss_type:       the type of loss function(Ritz, L2, Lncosh)      ------ string
             scale2lncosh:    if the loss is lncosh, using it                  ------- float
        return:
             UNN:             the output data
             loss_in:         the output loss in the interior points for given domain
        """
        assert (XYZ is not None)
        assert (fside is not None)

        shape2XYZ = XYZ.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        X = torch.reshape(XYZ[:, 0], shape=[-1, 1])
        Y = torch.reshape(XYZ[:, 1], shape=[-1, 1])
        Z = torch.reshape(XYZ[:, 2], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, Z)
        else:
            force_side = fside

        UNN = self.DNN(XYZ, scale=self.scales, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, XYZ, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0/2)*dUNN_2Norm-torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_in = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
            dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
            dUNN2z = torch.reshape(dUNN[:, 2], shape=[-1, 1])

            dUNNxxyz = torch.autograd.grad(dUNN2x, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNyxyz = torch.autograd.grad(dUNN2y, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNzxyz = torch.autograd.grad(dUNN2z, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNxx = torch.reshape(dUNNxxyz[:, 0], shape=[-1, 1])
            dUNNyy = torch.reshape(dUNNyxyz[:, 1], shape=[-1, 1])
            dUNNzz = torch.reshape(dUNNzxyz[:, 2], shape=[-1, 1])

            # -Laplace U = f --> -Laplace U - f --> -(Laplace U + f)
            loss_it_L2 = dUNNxx + dUNNyy + dUNNzz + torch.reshape(force_side, shape=[-1, 1])
            square_loss_it = torch.square(loss_it_L2)
            loss_in = torch.mean(square_loss_it)
        elif str.lower(loss_type) == 'l2_loss':
            dUNN2x = torch.reshape(dUNN[:, 0], shape=[-1, 1])
            dUNN2y = torch.reshape(dUNN[:, 1], shape=[-1, 1])
            dUNN2z = torch.reshape(dUNN[:, 2], shape=[-1, 1])

            dUNNxxyz = torch.autograd.grad(dUNN2x, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNyxyz = torch.autograd.grad(dUNN2y, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNzxyz = torch.autograd.grad(dUNN2z, XYZ, grad_outputs=torch.ones_like(X),
                                           create_graph=True, retain_graph=True)[0]
            dUNNxx = torch.reshape(dUNNxxyz[:, 0], shape=[-1, 1])
            dUNNyy = torch.reshape(dUNNyxyz[:, 1], shape=[-1, 1])
            dUNNzz = torch.reshape(dUNNzxyz[:, 2], shape=[-1, 1])

            # -Laplace U = f --> -Laplace U - f --> -(Laplace U + f)
            loss_it_temp = dUNNxx + dUNNyy + dUNNzz + torch.reshape(force_side, shape=[-1, 1])
            logcosh_loss_it = (1 / relaxation2lncosh) * torch.log(torch.cosh(relaxation2lncosh * loss_it_temp))
            loss_in = torch.mean(logcosh_loss_it)
        else:
            raise ValueError('loss type is not supported')
        return UNN, loss_in

    def loss2bd(self, XYZ_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='ritz_loss', relaxation2lncosh=0.1):
        """
        Calculating the loss of Laplace equation (*) or pLaplace equation with p==2 on the boundary points for given boundary
        BU = g             on Partial Omega, where B is a boundary operator, g is a given function
        Args:
            XY_bd:         the input data of variable. ------  float, shape=[B,D]
            Ubd_exact:     the exact function or array for boundary condition
            if_lambda2Ubd: the Ubd_exact is a lambda function or an array  ------ Bool
            loss_type:     the type of loss function(Ritz, L2, Lncosh)      ------ string
            scale2lncosh:  if the loss is lncosh, using it                  ------- float
        return:
            loss_bd: the output loss on the boundary points for given boundary
        """
        assert (XYZ_bd is not None)
        assert (Ubd_exact is not None)

        shape2XYZ = XYZ_bd.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)
        X_bd = torch.reshape(XYZ_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XYZ_bd[:, 1], shape=[-1, 1])
        Z_bd = torch.reshape(XYZ_bd[:, 2], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd, Z_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN(XYZ_bd, scale=self.scales, sFourier=self.sFourier)
        diff2bd = UNN_bd - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd_square = torch.mul(diff2bd, diff2bd)
            loss_bd = torch.mean(loss_bd_square)
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd_temp = (1 / relaxation2lncosh) * torch.log(torch.cosh(relaxation2lncosh * diff2bd))
            loss_bd = torch.mean(loss_bd_temp)
        else:
            raise ValueError('loss type is not supported')
        return loss_bd

    def get_regularSum2WB(self):
        """
        Calculating the regularization sum of weights and biases
        """
        sum2WB = self.DNN.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        return sum2WB

    def eval_model(self, XYZ_points=None):
        """
        Evaluating the MscaleDNN for testing points
        Args:
            XYZ_points: the testing input data of variable. ------  float, shape=[B,D]
        return:
            UNN: the testing output
        """
        assert (XYZ_points is not None)
        shape2XYZ = XYZ_points.shape
        lenght2XYZ_shape = len(shape2XYZ)
        assert (lenght2XYZ_shape == 2)
        assert (shape2XYZ[-1] == 3)

        UNN = self.DNN(XYZ_points, scale=self.scales, sFourier=self.sFourier)
        return UNN


def Solve_PDE(Rdic=None):
    log_out_path = Rdic['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log', Rdic['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(Rdic, log_fileout)

    # 问题需要的设置
    batchsize_in = Rdic['batch_size2interior']
    batchsize_bd = Rdic['batch_size2boundary']

    bd_penalty_init = Rdic['init_boundary_penalty']  # Regularization parameter for boundary conditions
    penalty2WB = Rdic['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = Rdic['learning_rate']

    input_dim = Rdic['input_dim']
    out_dim = Rdic['output_dim']

    # pLaplace 算子需要的额外设置, 先预设一下
    region_lb = 0.0
    region_rt = 1.0
    f_side, u_true, u_left, u_right, u_behind, u_front, u_bottom, u_top = General_Laplace.get_infos2PDE_3D(
        input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])

    model = PDE_DNN(input_dim=Rdic['input_dim'], out_dim=Rdic['output_dim'], hidden_layer=Rdic['hidden_layers'],
                    Model_name=Rdic['model_name'], name2actIn=Rdic['name2act_in'],
                    name2actHidden=Rdic['name2act_hidden'],
                    name2actOut=Rdic['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                    scales=Rdic['freq'], sFourier=Rdic['sfourier'], repeat_highFreq=Rdic['repeat_High_freq'],
                    use_gpu=Rdic['with_gpu'], No2GPU=Rdic['gpuNo'])
    if Rdic['with_gpu'] is True:
        model = model.cuda(device='cuda:' + str(Rdic['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)                # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    if Rdic['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # szie2test_data = 1600
        # size2test = 40
        szie2test_data = 4900
        size2test = 70
        # szie2test_data = 10000
        # size2test = 100
        test_xyz_torch = dataUtilizer2torch.rand_it(
            batch_size=szie2test_data, variable_dim=input_dim, region_a=region_lb, region_b=region_rt, to_torch=True,
            to_float=True, to_cuda=R['with_gpu'], gpu_no=R['gpuNo'], use_grad=False)
    elif str.lower(Rdic['testData_model']) == 'load_porous_domain_data':
        file2path = '../data2PorousDomain_3D/'
        test_xyz_torch = Load_data2Mat.get_MatData2Holes_3D(
            data_path=file2path, to_float=True, to_torch=True, to_cuda=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'],
            use_grad=False)
    else:
        file2path = '../data2RegularDomain_3D_FixedAxis/ThreeD2Fixed_Z'
        test_xyz_torch = Load_data2Mat.get_MatData2Regular_Domain3D(
            data_path=file2path, to_float=True, to_torch=True, to_cuda=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'],
            use_grad=False)

    Utrue2test = u_true(torch.reshape(test_xyz_torch[:, 0], shape=[-1, 1]),
                        torch.reshape(test_xyz_torch[:, 1], shape=[-1, 1]),
                        torch.reshape(test_xyz_torch[:, 2], shape=[-1, 1]))

    for i_epoch in range(Rdic['max_epoch'] + 1):
        # Generate randomly the training set(random or LatinHypercube) default=lhs
        xyz_it_batch = dataUtilizer2torch.rand_in_3D(
            batch_size=batchsize_in, variable_dim=Rdic['input_dim'], region_left=region_lb, region_right=region_rt,
            region_behind=region_lb, region_front=region_rt, region_bottom=region_lb, region_top=region_rt,
            to_torch=True, to_float=True, to_cuda=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'], use_grad=True)
        xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch, xyz_bottom_batch, xyz_top_batch = \
            dataUtilizer2torch.rand_bd_3D(
                batch_size=batchsize_bd, variable_dim=Rdic['input_dim'], region_left=region_lb, region_right=region_rt,
                region_behind=region_lb, region_front=region_rt, region_bottom=region_lb, region_top=region_rt,
                to_torch=True, to_float=True, to_cuda=Rdic['with_gpu'], gpu_no=Rdic['gpuNo'])

        if Rdic['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(Rdic['max_epoch'] / 10):
                temp_penalty_bd = bd_penalty_init
            elif i_epoch < int(Rdic['max_epoch'] / 5):
                temp_penalty_bd = 10 * bd_penalty_init
            elif i_epoch < int(Rdic['max_epoch'] / 4):
                temp_penalty_bd = 50 * bd_penalty_init
            elif i_epoch < int(Rdic['max_epoch'] / 2):
                temp_penalty_bd = 100 * bd_penalty_init
            elif i_epoch < int(3 * Rdic['max_epoch'] / 4):
                temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = 500 * bd_penalty_init
        else:
            temp_penalty_bd = bd_penalty_init

        UNN2train, loss_it = model.loss2in(XYZ=xyz_it_batch, fside=f_side, loss_type=Rdic['loss_type'],
                                           relaxation2lncosh=Rdic['scale2lncosh'])

        loss_bd2left = model.loss2bd(XYZ_bd=xyz_left_batch, Ubd_exact=u_left,
                                     loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd2right = model.loss2bd(XYZ_bd=xyz_right_batch, Ubd_exact=u_right,
                                      loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd2bottom = model.loss2bd(XYZ_bd=xyz_bottom_batch, Ubd_exact=u_bottom,
                                       loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd2top = model.loss2bd(XYZ_bd=xyz_top_batch, Ubd_exact=u_top,
                                    loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd2front = model.loss2bd(XYZ_bd=xyz_front_batch, Ubd_exact=u_front,
                                      loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd2behind = model.loss2bd(XYZ_bd=xyz_behind_batch, Ubd_exact=u_behind,
                                       loss_type=Rdic['loss_type2bd'], relaxation2lncosh=Rdic['scale2lncosh'])
        loss_bd = loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top + loss_bd2front + loss_bd2behind

        PWB = penalty2WB * model.get_regularSum2WB()

        loss = loss_it + temp_penalty_bd * loss_bd + PWB  # 要优化的loss function

        loss_it_all.append(loss_it.item())
        loss_bd_all.append(loss_bd.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()                 # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()                       # 对loss关于Ws和Bs求偏导
        optimizer.step()                      # 更新参数Ws和Bs
        scheduler.step()

        Uexact2train = u_true(torch.reshape(xyz_it_batch[:, 0], shape=[-1, 1]),
                              torch.reshape(xyz_it_batch[:, 1], shape=[-1, 1]),
                              torch.reshape(xyz_it_batch[:, 2], shape=[-1, 1]))
        train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
        train_rel = torch.sqrt(train_mse / torch.mean(torch.square(Uexact2train)))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            tmp_lr = optimizer.param_groups[0]['lr']
            run_times = time.time() - t0
            DNN_tools.print_and_log_train_one_epoch(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, PWB, loss_it.item(), loss_bd.item(), loss.item(),
                train_mse.item(), train_rel.item(), log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            test_epoch.append(i_epoch / 1000)

            UNN2test = model.eval_model(XYZ_points=test_xyz_torch)

            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = torch.sqrt(test_mse / torch.mean(torch.square(Utrue2test)))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=Rdic['name2act_hidden'],
                                         outPath=Rdic['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=Rdic['name2act_hidden'],
                                    outPath=Rdic['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_in', seedNo=Rdic['seed'], outPath=Rdic['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=Rdic['seed'], outPath=Rdic['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=Rdic['seed'], outPath=Rdic['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=Rdic['name2act_hidden'],
                                    outPath=Rdic['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=Rdic['name2act_hidden'],
                                         seedNo=Rdic['seed'], outPath=Rdic['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    if R['with_gpu'] is True:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        point_square_error_numpy = point_square_error.cpu().detach().numpy()
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()
        point_square_error_numpy = point_square_error.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, name2exact='utrue',
                                 name2dnn_solu=R['name2act_hidden'], outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=Rdic['name2act_hidden'],
                                  outPath=Rdic['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=Rdic['name2act_hidden'],
                              seedNo=Rdic['seed'], outPath=Rdic['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=Rdic['name2act_hidden'],
                                          outPath=Rdic['FolderName'])

    save_load_NetModule.save_torch_net2file_with_keys(
        outPath=Rdic['FolderName'], model2net=model, name2model='PoissonNN', optimizer=optimizer,
        scheduler=scheduler, epoch=Rdic['max_epoch'], loss=loss.item())


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    file2results = 'Results'
    store_file = 'Poisson3D'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    current_day_time = datetime.datetime.now()  # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')
    FolderName = os.path.join(OUT_DIR, date_time_dir)  # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    R['FolderName'] = FolderName
    R['seed'] = np.random.randint(1e5)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'Poisson'
    R['equa_name'] = 'PDE1'
    # R['equa_name'] = 'PDE2'

    R['mesh_number'] = 1
    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 750

    # ---------------------------- Setup of DNN -------------------------------
    # 装载测试数据模式
    # R['testData_model'] = 'random_generate'
    R['testData_model'] = 'load_regular_domain_data'
    # R['testData_model'] = 'load_irregular_domain_data'
    # R['testData_model'] = 'load_porous_domain_data'

    if R['testData_model'] == 'load_regular_domain_data':
        R['mesh_num'] = 7
        R['batch_size2test'] = 16384

    R['loss_type'] = 'L2_loss'                     # loss类型:L2 loss,  对应PINN 方法的loss
    # R['loss_type'] = 'Ritz_loss'                 # loss类型:PDE变分   对应 Deep Ritz method的loss
    # R['loss_type'] = 'lncosh_loss'

    R['loss_type2bd'] = 'l2_loss'

    if R['loss_type'] == 'lncosh_loss':
        R['loss_type2bd'] = 'lncosh_loss'

    # R['scale2lncosh'] = 0.01
    R['scale2lncosh'] = 0.05
    # R['scale2lncosh'] = 0.1
    # R['scale2lncosh'] = 0.5
    # R['scale2lncosh'] = 1

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 0.01  # 学习率
    # R['learning_rate'] = 0.005  # 学习率
    # R['learning_rate'] = 0.001  # 学习率
    # R['learning_rate'] = 2e-4   # 学习率

    R['scheduler2lr'] = 'StepLR'  # 学习率调整策略

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                      # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                   # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 1
    # R['init_boundary_penalty'] = 1000  # Regularization parameter for boundary conditions
    R['init_boundary_penalty'] = 20  # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    R['freq'] = np.arange(1, 30)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model_name'] = 'DNN'
    # R['model_name'] = 'Scale_DNN'
    # R['model_name'] = 'Fourier_DNN'
    R['model_name'] = 'Fourier_Sub_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model_name'] == 'Fourier_DNN':
        # R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
        R['hidden_layers'] = (225, 250, 200, 200, 150)  # 2*225+450*250+250*200+200*200+200*150+150*1=233100
        # R['hidden_layers'] = (50, 80, 60, 60, 40)
    elif R['model_name'] == 'Fourier_Sub_DNN':
        R['hidden_layers'] = (40, 20, 20, 10, 10)
    else:
        # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        R['hidden_layers'] = (250, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'enhance_tanh'
    # R['name2act_in'] = 's2relu'
    R['name2act_in'] = 'fourier'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'enhance_tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0
    # R['sfourier'] = 5.0
    # R['sfourier'] = 0.75

    R['with_gpu'] = True

    R['repeat_High_freq'] = True

    Solve_PDE(Rdic=R)

