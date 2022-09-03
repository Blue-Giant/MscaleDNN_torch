"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base
import DNN_tools
import dataUtilizer2torch
import MS_LaplaceEqs
import General_Laplace
import matData2HighDim
import saveData
import plotData


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    DNN_tools.log_string('Laplace name for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    DNN_tools.log_string('The order to p-laplace: %s\n' % (R_dic['order2laplace']), log_fileout)
    if R_dic['PDE_type'] == 'pLaplace':
        DNN_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)  # 替换上两行

    if R_dic['PDE_type'] == 'pLaplace':
        DNN_tools.log_string('The mesh_number: %f\n' % (R['mesh_number']), log_fileout)  # 替换上两行

    if R_dic['activate_stop'] != 0:
        DNN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)

    if R_dic['loss_type'] == 'variational_loss':
        DNN_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        DNN_tools.log_string('Loss function: L2 loss\n', log_fileout)

    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)
    DNN_tools.log_string('The frequency flags to Network: %s\n' % (R_dic['freqs']), log_fileout)
    DNN_tools.log_string('Activate function for network: %s\n' % str(R_dic['activate_func']), log_fileout)

    if R_dic['model'] == 'DNN_FourierBase' and R_dic['activate_func'] == 'tanh':
        DNN_tools.log_string('The scale-factor to fourier basis: %s\n' % (R_dic['sfourier']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        DNN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        DNN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    DNN_tools.log_string('Init learning rate: %s\n' % str(R_dic['learning_rate']), log_fileout)

    DNN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    DNN_tools.log_string('Batch-size 2 interior: %s\n' % str(R_dic['batch_size2interior']), log_fileout)
    DNN_tools.log_string('Batch-size 2 boundary: %s\n' % str(R_dic['batch_size2boundary']), log_fileout)

    DNN_tools.log_string('Initial boundary penalty: %s\n' % str(R_dic['init_boundary_penalty']), log_fileout)
    if R['activate_penalty2bd_increase'] == 1:
        DNN_tools.log_string('The penalty of boundary will increase with training going on.\n', log_fileout)
    elif R['activate_penalty2bd_increase'] == 2:
        DNN_tools.log_string('The penalty of boundary will decrease with training going on.\n', log_fileout)
    else:
        DNN_tools.log_string('The penalty of boundary will keep unchanged with training going on.\n', log_fileout)


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s%s.txt' % ('log2', R['activate_func'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    dictionary_out2file(R, log_fileout)

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

    # p laplace 问题需要的额外设置, 先预设一下
    p = 2
    mesh_number = 2

    region_lb = 0.0
    region_rt = 1.0
    if R['PDE_type'] == 'general_Laplace':
        # -laplace u = f
        region_lb = 0.0
        region_rt = 1.0
        f, u_true, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = General_Laplace.get_infos2Laplace_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=region_lb, intervalR=region_rt, equa_name=R['equa_name'])
    elif R['PDE_type'] == 'pLaplace':
        region_lb = 0.0
        region_rt = 1.0
        u_true, f, A_eps, u00, u01, u10, u11, u20, u21, u30, u31, u40, u41 = MS_LaplaceEqs.get_infos2pLaplace_5D(
            input_dim=input_dim, out_dim=out_dim, intervalL=0.0, intervalR=1.0, equa_name=R['equa_name'])

    flag = 'WB2NN'
    if R['model'] == 'DNN_FourierBase':
        W2NN, B2NN = DNN_base.Xavier_init_NN_Fourier(input_dim, out_dim, hidden_layers, flag)
    else:
        W2NN, B2NN = DNN_base.Xavier_init_NN(input_dim, out_dim, hidden_layers, flag)

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XYZST_it = tf.placeholder(tf.float32, name='XYZST_it', shape=[None, input_dim])
            XYZST00 = tf.placeholder(tf.float32, name='XYZST00', shape=[None, input_dim])
            XYZST01 = tf.placeholder(tf.float32, name='XYZST01', shape=[None, input_dim])
            XYZST10 = tf.placeholder(tf.float32, name='XYZST10', shape=[None, input_dim])
            XYZST11 = tf.placeholder(tf.float32, name='XYZST11', shape=[None, input_dim])
            XYZST20 = tf.placeholder(tf.float32, name='XYZST20', shape=[None, input_dim])
            XYZST21 = tf.placeholder(tf.float32, name='XYZST21', shape=[None, input_dim])
            XYZST30 = tf.placeholder(tf.float32, name='XYZST30', shape=[None, input_dim])
            XYZST31 = tf.placeholder(tf.float32, name='XYZST31', shape=[None, input_dim])
            XYZST40 = tf.placeholder(tf.float32, name='XYZST40', shape=[None, input_dim])
            XYZST41 = tf.placeholder(tf.float32, name='XYZST41', shape=[None, input_dim])
            boundary_penalty = tf.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')
            train_opt = tf.placeholder_with_default(input=True, shape=[], name='train_opt')

            # 供选择的网络模式
            if R['model'] == 'DNN':
                UNN = DNN_base.DNN(XYZST_it, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U00_NN = DNN_base.DNN(XYZST00, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U01_NN = DNN_base.DNN(XYZST01, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U10_NN = DNN_base.DNN(XYZST10, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U11_NN = DNN_base.DNN(XYZST11, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U20_NN = DNN_base.DNN(XYZST20, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U21_NN = DNN_base.DNN(XYZST21, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U30_NN = DNN_base.DNN(XYZST30, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U31_NN = DNN_base.DNN(XYZST31, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U40_NN = DNN_base.DNN(XYZST40, W2NN, B2NN, hidden_layers, activate_name=act_func)
                U41_NN = DNN_base.DNN(XYZST41, W2NN, B2NN, hidden_layers, activate_name=act_func)
            elif R['model'] == 'DNN_scale':
                freqs = R['freqs']
                UNN = DNN_base.DNN_scale(XYZST_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U00_NN = DNN_base.DNN_scale(XYZST00, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U01_NN = DNN_base.DNN_scale(XYZST01, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U10_NN = DNN_base.DNN_scale(XYZST10, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U11_NN = DNN_base.DNN_scale(XYZST11, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U20_NN = DNN_base.DNN_scale(XYZST20, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U21_NN = DNN_base.DNN_scale(XYZST21, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U30_NN = DNN_base.DNN_scale(XYZST30, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U31_NN = DNN_base.DNN_scale(XYZST31, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U40_NN = DNN_base.DNN_scale(XYZST40, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U41_NN = DNN_base.DNN_scale(XYZST41, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
            elif R['model'] == 'DNN_adapt_scale':
                freqs = R['freqs']
                UNN = DNN_base.DNN_adapt_scale(XYZST_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U00_NN = DNN_base.DNN_adapt_scale(XYZST00, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U01_NN = DNN_base.DNN_adapt_scale(XYZST01, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U10_NN = DNN_base.DNN_adapt_scale(XYZST10, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U11_NN = DNN_base.DNN_adapt_scale(XYZST11, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U20_NN = DNN_base.DNN_adapt_scale(XYZST20, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U21_NN = DNN_base.DNN_adapt_scale(XYZST21, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U30_NN = DNN_base.DNN_adapt_scale(XYZST30, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U31_NN = DNN_base.DNN_adapt_scale(XYZST31, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U40_NN = DNN_base.DNN_adapt_scale(XYZST40, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U41_NN = DNN_base.DNN_adapt_scale(XYZST41, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
            elif R['model'] == 'DNN_FourierBase':
                freqs = R['freqs']
                UNN = DNN_base.DNN_FourierBase(XYZST_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U00_NN = DNN_base.DNN_FourierBase(XYZST00, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U01_NN = DNN_base.DNN_FourierBase(XYZST01, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U10_NN = DNN_base.DNN_FourierBase(XYZST10, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U11_NN = DNN_base.DNN_FourierBase(XYZST11, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U20_NN = DNN_base.DNN_FourierBase(XYZST20, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U21_NN = DNN_base.DNN_FourierBase(XYZST21, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U30_NN = DNN_base.DNN_FourierBase(XYZST30, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U31_NN = DNN_base.DNN_FourierBase(XYZST31, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U40_NN = DNN_base.DNN_FourierBase(XYZST40, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
                U41_NN = DNN_base.DNN_FourierBase(XYZST41, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func, sFourier=R['sfourier'])
            elif R['model'] == 'DNN_Sin+Cos_Base':
                freqs = R['freqs']
                UNN = DNN_base.DNN_SinAddCos(XYZST_it, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U00_NN = DNN_base.DNN_SinAddCos(XYZST00, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U01_NN = DNN_base.DNN_SinAddCos(XYZST01, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U10_NN = DNN_base.DNN_SinAddCos(XYZST10, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U11_NN = DNN_base.DNN_SinAddCos(XYZST11, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U20_NN = DNN_base.DNN_SinAddCos(XYZST20, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U21_NN = DNN_base.DNN_SinAddCos(XYZST21, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U30_NN = DNN_base.DNN_SinAddCos(XYZST30, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U31_NN = DNN_base.DNN_SinAddCos(XYZST31, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U40_NN = DNN_base.DNN_SinAddCos(XYZST40, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)
                U41_NN = DNN_base.DNN_SinAddCos(XYZST41, W2NN, B2NN, hidden_layers, freqs, activate_name=act_func)

            X_it = tf.reshape(XYZST_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XYZST_it[:, 1], shape=[-1, 1])
            Z_it = tf.reshape(XYZST_it[:, 2], shape=[-1, 1])
            S_it = tf.reshape(XYZST_it[:, 3], shape=[-1, 1])
            T_it = tf.reshape(XYZST_it[:, 4], shape=[-1, 1])
            # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列, 因为 一个点对(x,y) 得到一个 u 值
            if R['loss_type'] == 'variational_loss':
                dUNN = tf.gradients(UNN, XYZST_it)[0]      # * 行 2 列
                dUNN_2norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
                if R['PDE_type'] == 'general_Laplace':
                    laplace_pow = tf.square(dUNN_2norm)
                    loss_it_variational = (1.0 / 2) *laplace_pow - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'pLaplace':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    laplace_p_pow = a_eps*tf.pow(dUNN_2norm, p)
                    if R['equa_name'] == 'multi_scale5D_4':
                        fxyzst = MS_LaplaceEqs.get_forceSide2pLaplace5D(x=X_it, y=Y_it, z=Z_it, s=S_it, t=T_it)
                        loss_it_variational = (1.0 / p) * laplace_p_pow - tf.multiply(tf.reshape(fxyzst, shape=[-1, 1]), UNN)
                    else:
                        loss_it_variational = (1.0 / p) * laplace_p_pow - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                elif R['PDE_type'] == 'Possion_Boltzmann':
                    a_eps = A_eps(X_it, Y_it, Z_it, S_it, T_it)                          # * 行 1 列
                    laplace_p_pow = a_eps*tf.pow(dUNN_2norm, p)
                    loss_it_variational = (1.0 / p) * laplace_p_pow - tf.multiply(f(X_it, Y_it, Z_it, S_it, T_it), UNN)
                loss_it = tf.reduce_mean(loss_it_variational)*(region_rt-region_lb)*(region_rt-region_lb)

                U_00 = tf.constant(0.0)
                U_01 = tf.constant(0.0)
                U_10 = tf.constant(0.0)
                U_11 = tf.constant(0.0)
                U_20 = tf.constant(0.0)
                U_21 = tf.constant(0.0)
                U_30 = tf.constant(0.0)
                U_31 = tf.constant(0.0)
                U_40 = tf.constant(0.0)
                U_41 = tf.constant(0.0)

                loss_bd_square2NN = tf.square(U00_NN - U_00) + tf.square(U01_NN - U_01) + tf.square(U10_NN - U_10) + \
                                    tf.square(U11_NN - U_11) + tf.square(U20_NN - U_20) + tf.square(U21_NN - U_21) + \
                                    tf.square(U30_NN - U_30) + tf.square(U31_NN - U_31) + tf.square(U40_NN - U_40) + \
                                    tf.square(U41_NN - U_41)
                loss_bd = tf.reduce_mean(loss_bd_square2NN)

            if R['regular_wb_model'] == 'L1':
                regularSum2WB = DNN_base.regular_weights_biases_L1(W2NN, B2NN)    # 正则化权重和偏置 L1正则化
            elif R['regular_wb_model'] == 'L2':
                regularSum2WB = DNN_base.regular_weights_biases_L2(W2NN, B2NN)    # 正则化权重和偏置 L2正则化
            else:
                regularSum2WB = tf.constant(0.0)                                        # 无正则化权重参数

            PWB = penalty2WB * regularSum2WB
            loss = loss_it + boundary_penalty * loss_bd + PWB      # 要优化的loss function

            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'group3_training':
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.group(train_op1, train_op2, train_op3)
            elif R['train_model'] == 'group2_training':
                train_op2bd = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op2union = my_optimizer.minimize(loss, global_step=global_steps)
                train_my_loss = tf.gruop(train_op2union, train_op2bd)
            elif R['train_model'] == 'union_training':
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            if R['PDE_type'] == 'general_Laplace' or R['PDE_type'] == 'pLaplace':
                # 训练上的真解值和训练结果的误差
                U_true = u_true(X_it, Y_it, Z_it, S_it, T_it)
                train_mse = tf.reduce_mean(tf.square(U_true - UNN))
                train_rel = train_mse / tf.reduce_mean(tf.square(U_true))
            else:
                train_mse = tf.constant(0.0)
                train_rel = tf.constant(0.0)

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
        test_xyzst_bach = DNN_data.rand_it(test_bach_size, input_dim, region_lb, region_rt)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])
    elif R['testData_model'] == 'loadData':
        test_bach_size = 1600
        size2test = 40
        mat_data_path = 'dataMat_highDim'
        test_xyzst_bach = matData2HighDim.get_data2Biharmonic(dim=input_dim, data_path=mat_data_path)
        saveData.save_testData_or_solus2mat(test_xyzst_bach, dataName='testXYZST', outPath=R['FolderName'])

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        train_option = True
        for i_epoch in range(R['max_epoch'] + 1):
            xyzst_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            xyzst00_batch, xyzst01_batch, xyzst10_batch, xyzst11_batch, xyzst20_batch, xyzst21_batch, xyzst30_batch, \
            xyzst31_batch, xyzst40_batch, xyzst41_batch = DNN_data.rand_bd_5D(batchsize_bd, input_dim,
                                                                              region_a=region_lb, region_b=region_rt)
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
            elif R['activate_penalty2bd_increase'] == 2:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = 5*bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 1 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 0.5 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 0.1 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 0.05 * bd_penalty_init
                else:
                    temp_penalty_bd = 0.02 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp, train_res_tmp, pwb = sess.run(
                [train_my_loss, loss_it, loss_bd, loss, train_mse, train_rel, PWB],
                feed_dict={XYZST_it: xyzst_it_batch, XYZST00: xyzst00_batch, XYZST01: xyzst01_batch,
                           XYZST10: xyzst10_batch, XYZST11: xyzst11_batch, XYZST20: xyzst20_batch,
                           XYZST21: xyzst21_batch, XYZST30: xyzst30_batch, XYZST31: xyzst31_batch,
                           XYZST40: xyzst40_batch, XYZST41: xyzst41_batch, in_learning_rate: tmp_lr,
                           boundary_penalty: temp_penalty_bd, train_opt: train_option})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_res_tmp)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_tools.print_and_log_train_one_epoch(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_tmp, train_mse_tmp,
                    train_res_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                train_option = False
                if R['PDE_type'] == 'general_laplace' or R['PDE_type'] == 'pLaplace':
                    u_true2test, u_nn2test = sess.run(
                        [U_true, UNN], feed_dict={XYZST_it: test_xyzst_bach, train_opt: train_option})
                else:
                    u_true2test = u_true
                    u_nn2test = sess.run(UNN,  feed_dict={XYZST_it: test_xyzst_bach, train_opt: train_option})

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

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 文件保存路径设置
    # store_file = 'Laplace5d'
    store_file = 'pLaplace5d'
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
        tf.compat.v1.reset_default_graph()
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
    R['input_dim'] = 5  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    if store_file == 'Laplace5d':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace5d':
        R['PDE_type'] = 'pLaplace'
        # R['equa_name'] = 'multi_scale5D_1'          # general laplace
        # R['equa_name'] = 'multi_scale5D_2'            # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_3'  # multi-scale laplace
        R['equa_name'] = 'multi_scale5D_4'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_5'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_6'  # multi-scale laplace
        # R['equa_name'] = 'multi_scale5D_7'  # multi-scale laplace

    if R['PDE_type'] == 'general_Laplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 12500          # 内部训练数据的批大小
        R['batch_size2boundary'] = 2000
    elif R['PDE_type'] == 'pLaplace':
        R['mesh_number'] = 1
        R['epsilon'] = 0.1
        R['order2laplace'] = 2
        R['batch_size2interior'] = 12500  # 内部训练数据的批大小
        R['batch_size2boundary'] = 2000

    # ---------------------------- Setup of DNN -------------------------------
    # 装载测试数据模式
    R['testData_model'] = 'loadData'
    # R['testData_model'] = 'random_generate'

    # R['loss_type'] = 'L2_loss'                             # loss类型:L2 loss
    R['loss_type'] = 'variational_loss'          # loss类型:PDE变分

    R['optimizer_name'] = 'Adam'                 # 优化器
    R['learning_rate'] = 2e-4                    # 学习率
    R['learning_rate_decay'] = 5e-5              # 学习率 decay
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
    # R['model'] = 'DNN_scale'
    # R['model'] = 'DNN_adapt_scale'
    R['model'] = 'DNN_FourierBase'
    # R['model'] = 'DNN_Sin+Cos_Base'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model'] == 'DNN_FourierBase':
        R['hidden_layers'] = (250, 400, 400, 300, 300, 200)  # 250+500*400+400*400+400*300+300*300+300*200+200=630450
    else:
        # R['hidden_layers'] = (100, 10, 8, 6, 4)  # 测试
        # R['hidden_layers'] = (100, 80, 60, 60, 40, 40, 20)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (250, 400, 400, 300, 300, 200)  # 250+500*400+400*400+400*300+300*300+300*200+200=630450
        R['hidden_layers'] = (500, 400, 400, 300, 300, 200)  # 500+500*400+400*400+400*300+300*300+300*200+200=630700
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 200, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['activate_func'] = 'relu'
    R['activate_func'] = 'tanh'
    # R['activate_func']' = leaky_relu'
    # R['activate_func'] = 'srelu'
    # R['activate_func'] = 's2relu'
    # R['activate_func'] = 'elu'
    # R['activate_func'] = 'phi'

    if R['model'] == 'DNN_FourierBase' and R['activate_func'] == 'tanh':
        R['sfourier'] = 0.5
        # R['sfourier'] = 1.0
    elif R['model'] == 'DNN_FourierBase' and R['activate_func'] == 's2relu':
        R['sfourier'] = 0.5

    solve_Multiscale_PDE(R)

