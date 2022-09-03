"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2022 年 9月 12 日
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
import DNN_base
import DNN_tools
import dataUtilizer2torch
import MS_LaplaceEqs
import MS_BoltzmannEqs
import General_Laplace
import Load_data2Mat
import saveData
import plotData
import DNN_Log_Print


class MscaleDNN(tn.Module):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, use_gpu=False, No2GPU=0):
        super(MscaleDNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
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

    def loss_it2Laplace(self, XY=None, fside=None, if_lambda2fside=True, loss_type='ritz_loss'):
        assert (XY is not None)
        assert (fside is not None)

        shape2XY = XY.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = torch.reshape(XY[:, 0], shape=[-1, 1])
        Y = torch.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y)
        else:
            force_side = fside

        UNN = self.DNN(XY, scale=self.factor2freq)
        grad2UNN = torch.autograd.grad(UNN, XY, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        dUNN = grad2UNN[0]

        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_2Norm = torch.reshape(torch.sum(torch.mul(dUNN, dUNN), dim=-1), shape=[-1, 1])  # 按行求和
            loss_it_ritz = (1.0/2)*dUNN_2Norm-torch.mul(torch.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = torch.mean(loss_it_ritz)
        elif str.lower(loss_type) == 'l2_loss':
            dUNN_x = torch.autograd.grad(dUNN[:, 0], XY, grad_outputs=torch.ones_like(XY),
                                         create_graph=True, retain_graph=True)[0]
            dUNN_y = torch.autograd.grad(dUNN[:, 1], XY, grad_outputs=torch.ones_like(XY),
                                         create_graph=True, retain_graph=True)[0]
            dUNNxxy = torch.autograd.grad(dUNN_x[:, 0], XY, grad_outputs=torch.ones_like(XY),
                                          create_graph=True, retain_graph=True)[0]
            dUNNyxy = torch.autograd.grad(dUNN_y[:, 1], XY, grad_outputs=torch.ones_like(XY),
                                          create_graph=True, retain_graph=True)[0]
            dUNNxx = dUNNxxy[:, 0]
            dUNNyy = dUNNyxy[:, 1]
            # -Laplace U=f --> -Laplace U - f --> -(Laplace U + f)
            loss_it_L2 = torch.add(dUNNxx, dUNNyy) + torch.reshape(force_side, shape=[-1, 1])
            square_loss_it = torch.mul(loss_it_L2, loss_it_L2)
            loss_it = torch.mean(square_loss_it)
        return UNN, loss_it

    def loss2bd(self, XY_bd=None, Ubd_exact=None, if_lambda2Ubd=True):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = XY_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X_bd = torch.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = torch.reshape(XY_bd[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN(XY_bd, scale=self.factor2freq)
        loss_bd_square = torch.mul(UNN_bd - Ubd, UNN_bd - Ubd)
        loss_bd = torch.mean(loss_bd_square)
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        UNN = self.DNN(XY_points, scale=self.factor2freq)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 一般 laplace 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']

    bd_penalty_init = R['init_boundary_penalty']                # Regularization parameter for boundary conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    hidden_layers = R['hidden_layers']
    act_func = R['activate_func']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    # pLaplace 算子需要的额外设置, 先预设一下
    p_index = 2
    mesh_number = 2

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u00, u01, u10, u11, u20, u21 = General_Laplace.get_infos2Laplace_3D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'Possion_Boltzmann':
        region_lb = 0.0
        region_rt = 1.0
        Aeps, kappa, f, u_true, u00, u01, u10, u11, u20, u21 = MS_BoltzmannEqs.get_infos2Boltzmann_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, Aeps, u00, u01, u10, u11, u20, u21 = MS_LaplaceEqs.get_infos2pLaplace_3D(
            input_dim=input_dim, out_dim=out_dim, mesh_number=R['mesh_number'], intervalL=0.0, intervalR=1.0,
            equa_name=R['equa_name'])

    mscalednn = MscaleDNN(input_dim=R['input_dim'], out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                          Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                          name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                          factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])
    if True == R['use_gpu']:
        mscalednn = mscalednn.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = mscalednn.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    # 画网格解图
    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        # test_bach_size = 400
        # size2test = 20
        # test_bach_size = 900
        # size2test = 30
        test_bach_size = 1600
        size2test = 40
        # test_bach_size = 4900
        # size2test = 70
        # test_bach_size = 10000
        # size2test = 100
        # test_bach_size = 40000
        # size2test = 200
        # test_bach_size = 250000
        # size2test = 500
        # test_bach_size = 1000000
        # size2test = 1000
        test_xyz_bach = dataUtilizer2torch.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyz_bach = dataUtilizer2torch.get_data2Biharmonic(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyz_bach, dataName='testXYZ', outPath=R['FolderName'])

    for i_epoch in range(R['max_epoch'] + 1):
        xyz_it_batch = dataUtilizer2torch.rand_it(batchsize_it, R['input_dim'], region_a=region_lb, region_b=region_rt,
                                             to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'],
                                             use_grad2x=True)
        xyz_bottom_batch, xyz_top_batch, xyz_left_batch, xyz_right_batch, xyz_front_batch, xyz_behind_batch = \
            dataUtilizer2torch.rand_bd_2D(batchsize_bd, R['input_dim'], region_a=region_lb, region_b=region_rt,
                                          to_float=True, to_cuda=R['use_gpu'], gpu_no=R['gpuNo'])
        tmp_lr = tmp_lr * (1 - lr_decay)
        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_bd = bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_bd = 10 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_bd = 50 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_bd = 100 * bd_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = 500 * bd_penalty_init
        else:
            temp_penalty_bd = bd_penalty_init

        if R['PDE_type'] == 'Laplace' or R['PDE_type'] == 'general_Laplace':
            UNN2train, loss_it = mscalednn.loss_it2Laplace(XY=xy_it_batch, fside=f, loss_type=R['loss_type'])

        loss_bd2left = mscalednn.loss2bd(XY_bd=xyz_left_batch, Ubd_exact=u_left)
        loss_bd2right = mscalednn.loss2bd(XY_bd=xyz_right_batch, Ubd_exact=u_right)
        loss_bd2bottom = mscalednn.loss2bd(XY_bd=xyz_bottom_batch, Ubd_exact=u_bottom)
        loss_bd2top = mscalednn.loss2bd(XY_bd=xyz_top_batch, Ubd_exact=u_top)
        loss_bd2front = mscalednn.loss2bd(XY_bd=xyz_front_batch, Ubd_exact=u_front)
        loss_bd2behind = mscalednn.loss2bd(XY_bd=xyz_behind_batch, Ubd_exact=u_behind)
        loss_bd = loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top + loss_bd2front + loss_bd2behind

        PWB = penalty2WB * mscalednn.get_regularSum2WB()

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
        train_rel = train_mse / torch.mean(torch.square(Uexact2train))

        train_mse_all.append(train_mse.item())
        train_rel_all.append(train_rel.item())

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            DNN_tools.print_and_log_train_one_epoch(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, PWB, loss_it.item(), loss_bd.item(), loss.item(),
                train_mse.item(), train_rel.item(), log_out=log_fileout)

            # ---------------------------   test network ----------------------------------------------
            test_epoch.append(i_epoch / 1000)
            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace' or \
                    R['PDE_type'] == 'Possion_Boltzmann':
                u_true2test, u_nn2test = sess.run(
                    [U_true, UNN], feed_dict={XYZ_it: test_xyz_bach, train_opt: train_option})
            else:
                u_true2test = u_true
                u_nn2test = sess.run(UNN,  feed_dict={XYZ_it: test_xyz_bach, train_opt: train_option})

            point_square_error = np.square(u_true2test - u_nn2test)
            mse2test = np.mean(point_square_error)
            test_mse_all.append(mse2test)
            res2test = mse2test / np.mean(np.square(u_true2test))
            test_rel_all.append(res2test)

            DNN_tools.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(u_true2test, u_nn2test, actName='utrue', actName1=act_func, outPath=R['FolderName'])

    # 绘制解的热力图(真解和DNN解)
    plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=size2test, actName='Utrue', seedNo=R['seed'],
                                    outPath=R['FolderName'])
    plotData.plot_Hot_solution2test(u_nn2test, size_vec2mat=size2test, actName=act_func, seedNo=R['seed'],
                                    outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])

    plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=size2test, actName=act_func,
                                     seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # 文件保存路径设置
    # store_file = 'Laplace3D'
    # store_file = 'pLaplace3D'
    store_file = 'Boltzmann3D'
    # store_file = 'Convection3D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
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

    # ---------------------------- Setup of multi-scale problem-------------------------------
    R['input_dim'] = 3  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    if store_file == 'Laplace3D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace3D':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale3D_1'
        # R['equa_name'] = 'multi_scale3D_2'
        R['equa_name'] = 'multi_scale3D_3'
    elif store_file == 'Boltzmann3D':
        R['PDE_type'] = 'Possion_Boltzmann'
        # R['equa_name'] = 'Boltzmann0'
        # R['equa_name'] = 'Boltzmann1'
        R['equa_name'] = 'Boltzmann2'
        # R['equa_name'] = 'Boltzmann3'
        # R['equa_name'] = 'Boltzmann4'
        # R['equa_name'] = 'Boltzmann5'
        # R['equa_name'] = 'Boltzmann6'
        # R['equa_name'] = 'Boltzmann7'
        # R['equa_name'] = 'Boltzmann8'
        # R['equa_name'] = 'Boltzmann9'
        # R['equa_name'] = 'Boltzmann10'

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000
    elif R['PDE_type'] == 'pLaplace'or R['PDE_type'] == 'Possion_Boltzmann':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
        R['batch_size2interior'] = 6000  # 内部训练数据的批大小
        R['batch_size2boundary'] = 1000

    # ---------------------------- Setup of DNN -------------------------------
    # 装载测试数据模式
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    # R['loss_type'] = 'L2_loss'                        # loss类型:L2 loss
    R['loss_type'] = 'variational_loss'                 # loss类型:PDE变分

    R['optimizer_name'] = 'Adam'                        # 优化器
    R['learning_rate'] = 2e-4                           # 学习率
    R['learning_rate_decay'] = 5e-5                     # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

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
    R['init_boundary_penalty'] = 100  # Regularization parameter for boundary conditions

    # 网络的频率范围设置
    R['freqs'] = np.concatenate(([1], np.arange(1, 100 - 1)), axis=0)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model'] = 'DNN'
    R['model'] = 'DNN_scale'
    # R['model'] = 'DNN_adapt_scale'
    # R['model'] = 'DNN_FourierBase'
    # R['model'] = 'DNN_Sin+Cos_Base'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model'] == 'DNN_FourierBase':
        R['hidden_layers'] = (250, 400, 400, 200, 200, 150)  # 250+500*400+400*400+400*200+200*200+200*150+150 = 510400
    else:
        # R['hidden_layers'] = (100, 10, 8, 6, 4)  # 测试
        # R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (300, 200, 150, 100, 100, 50, 50)
        R['hidden_layers'] = (500, 400, 400, 200, 200, 150)  # 500+500*400+400*400+400*200+200*200+200*150+150 = 510650
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100, 100)
        # R['hidden_layers'] = (500, 300, 200, 200, 100, 100, 50)
        # R['hidden_layers'] = (1000, 800, 600, 400, 200)
        # R['hidden_layers'] = (1000, 500, 400, 300, 300, 200, 100, 100)
        # R['hidden_layers'] = (2000, 1500, 1000, 500, 250)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['activate_func'] = 'relu'
    # R['activate_func'] = 'tanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    R['activate_func'] = 's2relu'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'phi'

    if R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'tanh':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 's2relu':
        R['sfourier'] = 0.5

    solve_Multiscale_PDE(R)

