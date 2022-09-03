import numpy as np
import Load_data2Mat


# 这里注意一下: 对于 np.ones_like(x), x要是一个有实际意义的树或数组或矩阵才可以。不可以是 tensorflow 占位符
# 如果x是占位符，要使用 np.ones_like
# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数

def get_infos2pLaplace1D(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps=0.01, equa_name=None):
    aeps = lambda x: 1.0 / (2 + np.cos(2 * np.pi * x / eps))

    utrue = lambda x: x - np.mul(x, x) + eps * (
            1 / np.pi * np.sin(np.pi * 2 * x / eps) * (1 / 4 - x / 2) - eps / (4 * np.pi ** 2) * np.cos(
        np.pi * 2 * x / eps) + eps / 4 / np.pi ** 2)

    u_l = lambda x: np.zeros_like(x)

    u_r = lambda x: np.zeros_like(x)

    if index2p == 2:
        f = lambda x: np.ones_like(x)
    elif index2p == 3:
        f = lambda x: abs(2 * x - 1) * (
                4 * eps + 2 * eps * np.cos(2 * np.pi * x / eps) + np.pi * (1 - 2 * x) * np.sin(2 * np.pi * x / eps)) / (
                              2 * eps)
    elif index2p == 4:
        f = lambda x: ((1 - 2 * x) ** 2) * (2 + np.cos(2 * np.pi * x / eps)) * (
                6 * eps + 3 * eps * np.cos(2 * np.pi * x / eps) - 2 * np.pi * (2 * x - 1) * np.sin(
            2 * np.pi * x / eps)) / (
                              4 * eps)
    elif index2p == 5:
        f = lambda x: -1.0 * abs((2 * x - 1) ** 3) * ((2 + np.cos(2 * np.pi * x / eps)) ** 2) * (
                3 * np.pi * (2 * x - 1) * np.sin(2 * np.pi * x / eps) - 4 * eps * np.cos(
            2 * np.pi * x / eps) - 8 * eps) / (
                              8 * eps)
    elif index2p == 8:
        f = lambda x: ((1 - 2 * x) ** 6) * ((2 + np.cos(2 * np.pi * x / eps)) ** 5) * (
                7 * eps * np.cos(2 * np.pi * x / eps) + 2 * (
                7 * eps - 3 * np.pi * (2 * x - 1) * np.sin(2 * np.pi * x / eps))) / (
                              64 * eps)
    else:

        f = lambda x: (np.power(abs(1 - 2 * x), index2p) * np.power(2 + np.cos(2 * np.pi * x / eps), index2p) * (
                eps * (index2p - 1) * (2 + np.cos(2 * np.pi * x / eps)) - np.pi * (index2p - 2) * (2 * x - 1) * np.sin(
            2 * np.pi * x / eps))) / (
                              np.power(2, index2p - 2) * eps * ((1 - 2 * x) ** 2) * (
                                  (2 + np.cos(2 * np.pi * x / eps)) ** 3))

    return utrue, f, aeps, u_l, u_r


def get_infos2pLaplace1D_3scale(in_dim=None, out_dim=None, intervalL=0, intervalR=1, index2p=2, eps1=0.1, eps2=0.01, equa_name=None):
    aeps = lambda x: (2 + np.cos(2 * np.pi * x / eps1)) * (2 + np.cos(2 * np.pi * x / eps2))

    utrue = lambda x: x - np.mul(x, x) + (eps1/(4*np.pi))*np.sin(2*np.pi*x/eps1) + (eps2/(4*np.pi))*np.sin(2*np.pi*x/eps2)

    u_l = lambda x: np.zeros_like(x)

    u_r = lambda x: np.zeros_like(x)

    f = lambda x: np.ones_like(x)

    return utrue, f, aeps, u_l, u_r


def force_side_3scale(x, eps1=0.02, eps2=0.01):
    aeps = (2 + np.cos(2 * np.pi * x / eps1)) * (2 + np.cos(2 * np.pi * x / eps2))

    aepsx = -(2 * np.pi / eps1) * np.sin(2 * np.pi * x / eps1) * (2 + np.cos(2 * np.pi * x / eps2)) - \
                   (2 * np.pi / eps2) * np.sin(2 * np.pi * x / eps2) * (2 + np.cos(2 * np.pi * x / eps1))

    ux = 1 - 2 * x + 0.5 * np.cos(2 * np.pi * x / eps1) + 0.5 * np.cos(2 * np.pi * x / eps2)

    uxx = -2 - (np.pi / eps1) * np.sin(2 * np.pi * x / eps1) - (np.pi / eps2) * np.sin(2 * np.pi * x / eps2)

    fside = -1.0*(aepsx * ux + aeps * uxx)

    return fside


#  例一
def true_solution2E1(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E1(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E1(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E1(input_dim=None, output_dim=None):
    a_eps = lambda x, y: 1.0*np.ones_like(x)
    return a_eps


#  例二
def true_solution2E2(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E2(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E2(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E2(input_dim=None, output_dim=None):
    a_eps = lambda x, y: 2.0 + np.multiply(np.sin(3 * np.pi * x), np.cos(5 * np.pi * y))
    return a_eps


# 例三
def true_solution2E3(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E3(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E3(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E3(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+np.sin(2*np.pi*x/e1))/(1.1+np.sin(2*np.pi*y/e1)) +
                              (1.1+np.sin(2*np.pi*y/e2))/(1.1+np.cos(2*np.pi*x/e2)) +
                              (1.1+np.cos(2*np.pi*x/e3))/(1.1+np.sin(2*np.pi*y/e3)) +
                              (1.1+np.sin(2*np.pi*y/e4))/(1.1+np.cos(2*np.pi*x/e4)) +
                              (1.1+np.cos(2*np.pi*x/e5))/(1.1+np.sin(2*np.pi*y/e5)) +
                              np.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例四
def true_solution2E4(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E4(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E4(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E4(input_dim=None, output_dim=None, mesh_num=2):
    if mesh_num == 2:
        a_eps = lambda x, y: (1+0.5*np.cos(2*np.pi*(x+y)))*(1+0.5*np.sin(2*np.pi*(y-3*x))) * \
                             (1+0.5*np.cos((2**2)*np.pi*(x+y)))*(1+0.5*np.sin((2**2)*np.pi*(y-3*x)))
    elif mesh_num==3:
        a_eps = lambda x, y: (1 + 0.5 * np.cos(2 * np.pi * (x + y))) * (1 + 0.5 * np.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * np.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 3) * np.pi * (y - 3 * x)))
    elif mesh_num == 4:
        a_eps = lambda x, y: (1 + 0.5 * np.cos(2 * np.pi * (x + y))) * (1 + 0.5 * np.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * np.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 4) * np.pi * (y - 3 * x)))
    elif mesh_num == 5:
        a_eps = lambda x, y: (1 + 0.5 * np.cos(2 * np.pi * (x + y))) * (1 + 0.5 * np.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * np.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * np.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 5) * np.pi * (y - 3 * x)))
    elif mesh_num == 6:
        a_eps = lambda x, y: (1 + 0.5 * np.cos(2 * np.pi * (x + y))) * (1 + 0.5 * np.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * np.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * np.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 5) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * np.cos((2 ** 6) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 6) * np.pi * (y - 3 * x)))
    elif mesh_num == 7:
        a_eps = lambda x, y: (1 + 0.5 * np.cos(2 * np.pi * (x + y))) * (1 + 0.5 * np.sin(2 * np.pi * (y - 3 * x))) * \
                             (1 + 0.5 * np.cos((2 ** 2) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 2) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 3) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 3) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 4) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 4) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * np.cos((2 ** 5) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 5) * np.pi * (y - 3 * x))) \
                             * (1 + 0.5 * np.cos((2 ** 6) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 6) * np.pi * (y - 3 * x)))\
                             * (1 + 0.5 * np.cos((2 ** 7) * np.pi * (x + y))) * (1 + 0.5 * np.sin((2 ** 7) * np.pi * (y - 3 * x)))
    return a_eps


# 例五
def true_solution2E5(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E5(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E5(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E5(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+np.sin(2*np.pi*x/e1))/(1.1+np.sin(2*np.pi*y/e1)) +
                              (1.1+np.sin(2*np.pi*y/e2))/(1.1+np.cos(2*np.pi*x/e2)) +
                              (1.1+np.cos(2*np.pi*x/e3))/(1.1+np.sin(2*np.pi*y/e3)) +
                              (1.1+np.sin(2*np.pi*y/e4))/(1.1+np.cos(2*np.pi*x/e4)) +
                              (1.1+np.cos(2*np.pi*x/e5))/(1.1+np.sin(2*np.pi*y/e5)) +
                              np.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例六
def true_solution2E6(input_dim=None, output_dim=None, q=2, file_name=None):
    mat_true = Load_data2Mat.load_Matlab_data(filename=file_name)
    true_key = 'u_true'
    utrue = mat_true[true_key]
    return utrue


def force_side2E6(input_dim=None, output_dim=None):
    f_side = lambda x, y: 1.0*np.ones_like(x)
    return f_side


def boundary2E6(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0):
    # ux_left = lambda x, y: np.exp(-left_bottom)*(np.pow(y, 3) + 1.0*left_bottom)
    # ux_right = lambda x, y: np.exp(-right_top)*(np.pow(y, 3) + 1.0*right_top)
    # uy_bottom = lambda x, y: np.exp(-x)*(np.pow(left_bottom, 3) + x)
    # uy_top = lambda x, y: np.exp(-x)*(np.pow(right_top, 3) + x)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E6(input_dim=None, output_dim=None):
    e1 = 1.0 / 5
    e2 = 1.0 / 13
    e3 = 1.0 / 17
    e4 = 1.0 / 31
    e5 = 1.0 / 65
    a_eps = lambda x, y: (1.0/6)*((1.1+np.sin(2*np.pi*x/e1))/(1.1+np.sin(2*np.pi*y/e1)) +
                              (1.1+np.sin(2*np.pi*y/e2))/(1.1+np.cos(2*np.pi*x/e2)) +
                              (1.1+np.cos(2*np.pi*x/e3))/(1.1+np.sin(2*np.pi*y/e3)) +
                              (1.1+np.sin(2*np.pi*y/e4))/(1.1+np.cos(2*np.pi*x/e4)) +
                              (1.1+np.cos(2*np.pi*x/e5))/(1.1+np.sin(2*np.pi*y/e5)) +
                              np.sin(4*(x**2)*(y**2))+1)
    return a_eps


# 例七
def true_solution2E7(input_dim=None, output_dim=None, eps=0.1):
    # utrue = lambda x, y: 0.5*np.sin(np.pi*x)*np.sin(np.pi*y)+0.025*np.sin(10*np.pi*x)*np.sin(10*np.pi*y)
    utrue = lambda x, y: 1.0*np.sin(np.pi*x) * np.sin(np.pi*y) + 0.05 * np.sin(20*np.pi * x) * np.sin(20*np.pi*y)
    return utrue


def get_force_side2MS_E7(x=None, y=None, input_dim=None, output_dim=None):
    # utrue = np.sin(np.pi * x) * np.sin(np.pi * y) + 0.05 * np.sin(20 * np.pi * x) * np.sin( 20 * np.pi * y)
    Aeps = 0.5 + 0.125*np.cos(10.0*np.pi*x)*np.cos(10.0*np.pi*y) + 0.125*np.cos(20.0*np.pi*x)*np.cos(20.0*np.pi*y)

    ux = (np.pi)*np.cos(np.pi*x)*np.sin(np.pi*y) + (np.pi)*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)
    uy = (np.pi)*np.sin(np.pi*x)*np.cos(np.pi*y) + (np.pi)*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)

    uxx = -1.0*((np.pi)**2)*np.sin(np.pi*x)*np.sin(np.pi*y) - 20.0*((np.pi)**2)*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

    uyy = -1.0*((np.pi)**2)*np.sin(np.pi*x)*np.sin(np.pi*y) - 20.0*((np.pi)**2)*np.sin(20*np.pi*x)*np.sin(20*np.pi*y)

    # unorm = np.sqrt(np.square(ux)+np.square(uy))

    Aepsx = -1.25*np.pi*np.sin(10*np.pi*x)*np.cos(10*np.pi*y) - 2.5*np.pi*np.sin(20*np.pi*x)*np.cos(20*np.pi*y)

    Aepsy = -1.25*np.pi*np.cos(10*np.pi*x)*np.sin(10*np.pi*y) - 2.5*np.pi*np.cos(20*np.pi*x)*np.sin(20*np.pi*y)

    fside = -1.0*(Aepsx*ux + Aeps*uxx + Aepsy*uy + Aeps*uyy)

    return fside


def boundary2E7(input_dim=None, output_dim=None, left_bottom=0.0, right_top=1.0, eps=0.1):
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)
    return ux_left, ux_right, uy_bottom, uy_top


def elliptic_coef2E7(input_dim=None, output_dim=None, eps=0.1):
    a_eps = lambda x, y: 0.5 + 0.125*np.cos(10*np.pi*x)*np.cos(10.0*np.pi*y) + \
                         0.125*np.cos(20.0*np.pi*x)*np.cos(20.0*np.pi*y)
    return a_eps


def get_infos2pLaplace_2D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale2D_1':
        f = force_side2E1(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E1/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E1(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E1(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E1(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_2':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E2(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E2/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E2(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E2(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E2(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_3':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E3(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E3/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E3(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E3(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E3(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_4':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E4(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E4/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E4(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E4(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E4(input_dim, out_dim, mesh_num=mesh_number)
    elif equa_name == 'multi_scale2D_5':
        intervalL = 0
        intervalR = 1.0
        f = force_side2E5(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E5/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E5(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E5(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E5(input_dim, out_dim)
    elif equa_name == 'multi_scale2D_6':
        intervalL = -1.0
        intervalR = 1.0
        f = force_side2E6(input_dim, out_dim)  # f是一个向量
        u_true_filepanp = 'dataMat2pLaplace/E6/' + str('u_true') + str(mesh_number) + str('.mat')
        u_true = true_solution2E6(input_dim, out_dim, q=mesh_number, file_name=u_true_filepanp)
        u_left, u_right, u_bottom, u_top = boundary2E6(input_dim, out_dim, intervalL, intervalR)
        # A_eps要作用在u的每一个网格点值，所以A_eps在每一个网格点都要求值，和u类似
        A_eps = elliptic_coef2E6(input_dim, out_dim)

    return u_true, f, A_eps, u_left, u_right, u_bottom, u_top


def get_infos2pLaplace_3D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale3D_1':
        fside = lambda x, y, z: 3.0 * ((np.pi)**2) * (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
        u_true = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0
        u_00 = lambda x, y, z: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_01 = lambda x, y, z: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_10 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z)
        u_11 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z)
        u_20 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_2':
        fside = lambda x, y, z: 3.0*((np.pi)**2)*(1.0+np.cos(np.pi*x)*np.cos(3*np.pi*y)*np.cos(5*np.pi*z))*\
                                (np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z))+\
        ((np.pi)**2)*(np.sin(np.pi*x)*np.cos(3*np.pi*y)*np.cos(5*np.pi*z)*np.cos(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z))+\
        3.0*((np.pi)** 2)*(np.cos(np.pi*x)*np.sin(3*np.pi*y)*np.cos(5*np.pi*z)*np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z))+ \
        5.0*((np.pi)**2)*(np.cos(np.pi*x)*np.cos(3*np.pi*y)*np.sin(5*np.pi*z)*np.sin(np.pi * x)*np.sin(np.pi*y)*np.cos(np.pi*z))
        u_true = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0 + np.cos(np.pi * x) * np.cos(3 * np.pi * y) * np.cos(5 * np.pi * z)
        u_00 = lambda x, y, z: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_01 = lambda x, y, z: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_10 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z)
        u_11 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z)
        u_20 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale3D_3':
        fside = lambda x, y, z: (63/4)*((np.pi)**2)*(1.0+np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z))*\
                                (np.sin(np.pi*x) * np.sin(5*np.pi*y) * np.sin(10*np.pi*z))+\
        0.125*((np.pi)**2)*np.sin(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(np.pi*x)*np.sin(5*np.pi*y)*np.sin(10*np.pi*z)+\
        (25/4)*((np.pi)** 2)*np.cos(np.pi*x)*np.sin(10*np.pi*y)*np.cos(20*np.pi*z)*np.sin(np.pi*x)*np.cos(5*np.pi*y)*np.sin(10*np.pi*z)+ \
        25.0*((np.pi)**2)*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(np.pi * x)*np.sin(5*np.pi*y)*np.cos(10*np.pi*z)
        u_true = lambda x, y, z: 0.5*np.sin(np.pi * x) * np.sin(5*np.pi * y) * np.sin(10*np.pi * z)
        A_eps = lambda x, y, z: 0.25*(1.0 + np.cos(np.pi * x) * np.cos(10 * np.pi * y) * np.cos(20 * np.pi * z))
        u_00 = lambda x, y, z: 0.5*np.sin(np.pi * intervalL) * np.sin(5*np.pi * y) * np.sin(np.pi * z)
        u_01 = lambda x, y, z: 0.5*np.sin(np.pi * intervalR) * np.sin(5*np.pi * y) * np.sin(np.pi * z)
        u_10 = lambda x, y, z: 0.5*np.sin(np.pi * x) * np.sin(5*np.pi * intervalL) * np.sin(np.pi * z)
        u_11 = lambda x, y, z: 0.5*np.sin(np.pi * x) * np.sin(5*np.pi * intervalR) * np.sin(np.pi * z)
        u_20 = lambda x, y, z: 0.5*np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10*np.pi * intervalL)
        u_21 = lambda x, y, z: 0.5*np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(10*np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21


def get_infos2pLaplace_4D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale4D_1':
        fside = lambda x, y, z: 3.0 * ((np.pi)**2) * (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
        u_true = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0
        u_00 = lambda x, y, z: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_01 = lambda x, y, z: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z)
        u_10 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z)
        u_11 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z)
        u_20 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21
    elif equa_name == 'multi_scale4D_2':
        fside = lambda x, y, z, s: np.ones_like(s)
        u_true = lambda x, y, z, s: np.sin(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.sin(10.0*np.pi*z)*np.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 0.25*(1.0+np.cos(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.cos(10.0*np.pi*z)*np.cos(5.0*np.pi*s))
        u_00 = lambda x, y, z, s: np.sin(5*np.pi * intervalL) * np.sin(10*np.pi * y) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: np.sin(5*np.pi * intervalR) * np.sin(10*np.pi * y) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_10 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalR) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_20 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(10*np.pi * intervalL) * np.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(10*np.pi * intervalR) * np.sin(5*np.pi * s)
        u_30 = lambda x, y, z, s: np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31
    elif equa_name == 'multi_scale4D_3':
        fside = lambda x, y, z, s: np.ones_like(s)
        u_true = lambda x, y, z, s: np.sin(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.sin(10.0*np.pi*z)*np.sin(5.0*np.pi*s)
        A_eps = lambda x, y, z, s: 0.25*(1.0+np.cos(5.0*np.pi*x)*np.cos(20.0*np.pi*y)*np.cos(20.0*np.pi*z)*np.cos(5.0*np.pi*s))
        u_00 = lambda x, y, z, s: np.sin(5*np.pi * intervalL) * np.sin(10*np.pi * y) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_01 = lambda x, y, z, s: np.sin(5*np.pi * intervalR) * np.sin(10*np.pi * y) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_10 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_11 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalR) * np.sin(10*np.pi * z) * np.sin(5*np.pi * s)
        u_20 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(10*np.pi * intervalL) * np.sin(5*np.pi * s)
        u_21 = lambda x, y, z, s: np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(10*np.pi * intervalR) * np.sin(5*np.pi * s)
        u_30 = lambda x, y, z, s: np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * intervalL)
        u_31 = lambda x, y, z, s: np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31


def get_force2pLaplace_4D(x=None, y=None, z=None, s=None, equa_name=None):
    if equa_name == 'multi_scale4D_2':
        A = 0.25*(1.0 + np.cos(5.0*np.pi * x)*np.cos(10.0*np.pi*y) * np.cos(10.0 *np.pi*z) * np.cos(5.0*np.pi * s))
        Ax = -0.25 *5.0*np.pi*np.sin(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.cos(5.0*np.pi*s)
        Ay = -0.25*10.0*np.pi*np.cos(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.cos(5.0* np.pi * s)
        Az = -0.25*10.0*np.pi*np.cos(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.sin(10.0*np.pi*z) * np.cos(5.0* np.pi * s)
        As = -0.25 *5.0*np.pi*np.cos(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.sin(5.0* np.pi * s)
        U = np.sin(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.sin(10.0*np.pi*z)*np.sin(5.0*np.pi*s)
        Ux = 5*np.pi * np.cos(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Uy = 10*np.pi* np.sin(5 * np.pi * x) * np.cos(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Uz = 10*np.pi* np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.cos(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Us = 5*np.pi * np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.cos(5 * np.pi * s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) + 250.0*A*np.pi*np.pi*U
        return fside
    elif equa_name == 'multi_scale4D_3':
        A = 0.25*(1.0 + np.cos(5.0*np.pi * x)*np.cos(20.0*np.pi*y) * np.cos(20.0 *np.pi*z) * np.cos(5.0*np.pi * s))
        Ax = -0.25 *5.0*np.pi*np.sin(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.cos(5.0*np.pi*s)
        Ay = -0.25*20.0*np.pi*np.cos(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.cos(5.0* np.pi * s)
        Az = -0.25*20.0*np.pi*np.cos(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.sin(10.0*np.pi*z) * np.cos(5.0* np.pi * s)
        As = -0.25 *5.0*np.pi*np.cos(5.0*np.pi*x)*np.cos(10.0*np.pi*y)*np.cos(10.0*np.pi*z) * np.sin(5.0* np.pi * s)
        U = np.sin(5.0*np.pi*x)*np.sin(10.0*np.pi*y)*np.sin(10.0*np.pi*z)*np.sin(5.0*np.pi*s)
        Ux = 5*np.pi * np.cos(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Uy = 10*np.pi* np.sin(5 * np.pi * x) * np.cos(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Uz = 10*np.pi* np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.cos(10 * np.pi * z) * np.sin(5 * np.pi * s)
        Us = 5*np.pi * np.sin(5 * np.pi * x) * np.sin(10 * np.pi * y) * np.sin(10 * np.pi * z) * np.cos(5 * np.pi * s)

        fside = -1.0*(Ax*Ux + Ay*Uy + Az*Uz + As*Us) + 250.0*A*np.pi*np.pi*U
        return fside


def get_infos2pLaplace_5D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale5D_1':
        fside = lambda x, y, z, s, t: 5.0 * ((np.pi) ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)* np.sin(np.pi * s)* np.sin(np.pi * t)
        u_true = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)* np.sin(np.pi * s)* np.sin(np.pi * t)
        A_eps = lambda x, y, z, s, t: 1.0
        u_00 = lambda x, y, z, s, t: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR) * np.sin(np.pi * s) * np.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalL) * np.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalR) * np.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_2':
        u_true = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)
        fside = lambda x, y, z, s, t: 5.0*((np.pi)**2)*np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)\
                                      *(1.0+np.cos(np.pi*x)*np.cos(2*np.pi*y)*np.cos(3*np.pi*z)*np.cos(2*np.pi*s)*np.cos(np.pi*t))\
                                      +((np.pi)**2)*np.sin(np.pi*x)*np.cos(2*np.pi*y)*np.cos(3*np.pi*z)*np.cos(2*np.pi*s)*np.cos(np.pi*t)\
                                      *np.cos(np.pi * x) * np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)\
                                      +(2.0*(np.pi)**2)*np.cos(np.pi*x)*np.sin(2*np.pi*y)*np.cos(3*np.pi*z)*np.cos(2*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.cos(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)\
                                      +(3.0*(np.pi)** 2)*np.cos(np.pi*x)*np.cos(2*np.pi*y)*np.sin(3*np.pi*z)*np.cos(2*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.sin(np.pi*y)*np.cos(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)\
                                      +(2.0*(np.pi)**2)*np.cos(np.pi*x)*np.cos(2*np.pi*y)*np.cos(3*np.pi*z)*np.sin(2*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.cos(np.pi*s)*np.sin(np.pi*t)\
                                      +((np.pi)**2)*np.cos(np.pi*x)*np.cos(2*np.pi*y)*np.cos(3*np.pi*z)*np.cos(2*np.pi*s)*np.sin(np.pi*t) \
                                      *np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.cos(np.pi*t)
        A_eps = lambda x, y, z, s, t: 1.0 + np.cos(np.pi*x)*np.cos(2*np.pi*y)*np.cos(3*np.pi*z)*np.cos(2*np.pi*s)*np.cos(np.pi*t)
        u_00 = lambda x, y, z, s, t: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL) * np.sin(
            np.pi * s) * np.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR) * np.sin(
            np.pi * s) * np.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * intervalL) * np.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * intervalR) * np.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(
            np.pi * s) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_3':
        u_true = lambda x, y, z, s, t: 0.5*np.sin(np.pi * x) * np.sin(5*np.pi * y) * np.sin(10*np.pi * z)* np.sin(5*np.pi * s)* np.sin(np.pi * t)
        fside = lambda x, y, z, s, t: 19*((np.pi)**2)*np.sin(np.pi*x)*np.sin(5*np.pi*y)*np.sin(10*np.pi*z)*np.sin(5*np.pi*s)*np.sin(np.pi*t)\
                                      *(1.0+np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t))\
                                      +0.125*((np.pi)**2)*np.sin(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)\
                                      *np.cos(np.pi * x) * np.sin(5*np.pi*y)*np.sin(10*np.pi*z)*np.sin(5*np.pi*s)*np.sin(np.pi*t)\
                                      +6.25*((np.pi)**2)*np.cos(np.pi*x)*np.sin(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.cos(5*np.pi*y)*np.sin(10*np.pi*z)*np.sin(5*np.pi*s)*np.sin(np.pi*t)\
                                      +25*((np.pi)** 2)*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.sin(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.sin(5*np.pi*y)*np.cos(10*np.pi*z)*np.sin(5*np.pi*s)*np.sin(np.pi*t)\
                                      +6.25*((np.pi)**2)*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.sin(10*np.pi*s)*np.cos(np.pi*t)\
                                      *np.sin(np.pi*x)*np.sin(5*np.pi*y)*np.sin(10*np.pi*z)*np.cos(5*np.pi*s)*np.sin(np.pi*t)\
                                      +0.125*((np.pi)**2)*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.sin(np.pi*t) \
                                      *np.sin(np.pi*x)*np.sin(5*np.pi*y)*np.sin(10*np.pi*z)*np.sin(5*np.pi*s)*np.cos(np.pi*t)
        A_eps = lambda x, y, z, s, t: 0.25*(1.0 + np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t))
        u_00 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * intervalL) * np.sin(5 * np.pi * y) * \
                                     np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s) * np.sin(np.pi * t)
        u_01 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * intervalR) * np.sin(5 * np.pi * y) * \
                                     np.sin(10 * np.pi * z) * np.sin(5 * np.pi * s) * np.sin(np.pi * t)

        u_10 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * intervalL) * np.sin(10 * np.pi * z) \
                                     * np.sin(5 * np.pi * s) * np.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * intervalR) * np.sin(10 * np.pi * z) \
                                     * np.sin(5 * np.pi * s) * np.sin(np.pi * t)

        u_20 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * intervalL) \
                                     * np.sin(5 * np.pi * s) * np.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * intervalR) \
                                     * np.sin(5 * np.pi * s) * np.sin(np.pi * t)

        u_30 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z) * \
                                     np.sin(5 * np.pi * intervalL) * np.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z) * \
                                     np.sin(5 * np.pi * intervalR) * np.sin(np.pi * t)

        u_40 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z) * \
                                     np.sin(5 * np.pi * s) * np.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: 0.5 * np.sin(np.pi * x) * np.sin(5 * np.pi * y) * np.sin(10 * np.pi * z) * \
                                     np.sin(np.pi * s) * np.sin(np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_4':
        u_true = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        fside = lambda x, y, z, s, t: np.oneslike(x)
        A_eps = lambda x, y, z, s, t: 0.25*(1.0+np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t))
        u_00 = lambda x, y, z, s, t: np.sin(np.pi*intervalL)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        u_01 = lambda x, y, z, s, t: np.sin(np.pi*intervalR)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)

        u_10 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*intervalL)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        u_11 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*intervalR)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)

        u_20 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*intervalL)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        u_21 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*intervalR)*np.sin(10*np.pi*s)*np.sin(np.pi*t)

        u_30 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*intervalL)*np.sin(np.pi*t)
        u_31 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*intervalR)*np.sin(np.pi*t)

        u_40 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*intervalL)
        u_41 = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_5':
        u_true = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        fside = lambda x, y, z, s, t: 5*((np.pi)**2)*np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       3.25*((np.pi)**2)*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        A_eps = lambda x, y, z, s, t: np.ones_like(x)

        u_00 = lambda x, y, z, s, t: np.sin(np.pi * intervalL) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * intervalL) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_01 = lambda x, y, z, s, t: np.sin(np.pi * intervalR) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * intervalR) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_10 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_11 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalR) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_20 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_21 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR) * np.sin(np.pi * s) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * intervalR)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_30 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalL) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_31 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalR) * np.sin(np.pi * t)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * intervalR)* np.sin(5*np.pi * t)

        u_40 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalL)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_41 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalR)+ \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_6':
        u_true = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z) * np.sin(10*np.pi * s) * np.sin(5*np.pi * t)
        fside = lambda x, y, z, s, t: 100 * (y - np.square(y)) * (z - np.square(z)) * (s - np.square(s)) * (t - np.square(t)) + \
                                      100 * (x - np.square(x)) * (z - np.square(z)) * (s - np.square(s)) * (t - np.square(t)) + \
                                      100 * (x - np.square(x)) * (y - np.square(y)) * (s - np.square(s)) * (t - np.square(t)) + \
                                      100 * (x - np.square(x)) * (y - np.square(y)) * (y - np.square(y)) * (t - np.square(t)) + \
                                      100 * (x - np.square(x)) * (y - np.square(y)) * (z - np.square(z)) * (s - np.square(s)) + \
                                      6.5*((np.pi)**2)*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        A_eps = lambda x, y, z, s, t: np.ones_like(x)

        u_00 = lambda x, y, z, s, t: 50*(intervalL - np.square(intervalL))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * intervalL) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_01 = lambda x, y, z, s, t: 50*(intervalR - np.square(intervalR))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * intervalR) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_10 = lambda x, y, z, s, t: 50*(x - np.square(x))*(intervalL - np.square(intervalL))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_11 = lambda x, y, z, s, t: 50*(x - np.square(x))*(intervalR - np.square(intervalR))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalR) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_20 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(intervalL - np.square(intervalL))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_21 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(intervalR - np.square(intervalR))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * intervalR)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_30 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(intervalL - np.square(intervalL))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_31 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(intervalR - np.square(intervalR))*(t - np.square(t)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * intervalR)* np.sin(5*np.pi * t)

        u_40 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(s - np.square(intervalL))*(t - np.square(intervalL)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_41 = lambda x, y, z, s, t: 50*(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(intervalR - np.square(intervalR)) + \
                                       0.1*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_7':
        u_true = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))*(1-np.cos(2*np.pi*s))*(1-np.cos(2*np.pi*t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z) * np.sin(10*np.pi * s) * np.sin(5*np.pi * t)
        fside = lambda x, y, z, s, t: -(4*(np.pi*x)**2)*np.cos(2*np.pi*x)*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))*(1-np.cos(2*np.pi*s))*(1-np.cos(2*np.pi*t)) - \
                                      (4*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*np.cos(2*np.pi*y)*(1-np.cos(2*np.pi*z))*(1-np.cos(2*np.pi*s))*(1-np.cos(2*np.pi*t)) - \
                                      (4*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*np.cos(2*np.pi*z)*(1-np.cos(2*np.pi*s))*(1-np.cos(2*np.pi*t)) - \
                                      (4*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))*np.cos(2*np.pi*s)*(1-np.cos(2*np.pi*t)) - \
                                      (4*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(2*np.pi*y))*(1-np.cos(2*np.pi*z))*(1-np.cos(2*np.pi*s))*np.cos(2*np.pi*t) + \
                                      3.25*((np.pi)**2)*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        A_eps = lambda x, y, z, s, t: np.ones_like(x)

        u_00 = lambda x, y, z, s, t: (intervalL - np.square(intervalL))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * intervalL) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_01 = lambda x, y, z, s, t: (intervalR - np.square(intervalR))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * intervalR) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_10 = lambda x, y, z, s, t: (x - np.square(x))*(intervalL - np.square(intervalL))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_11 = lambda x, y, z, s, t: (x - np.square(x))*(intervalR - np.square(intervalR))*(z - np.square(z))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalR) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_20 = lambda x, y, z, s, t: (x - np.square(x))*(y - np.square(y))*(intervalL - np.square(intervalL))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_21 = lambda x, y, z, s, t: (x - np.square(x))*(y - np.square(y))*(intervalR - np.square(intervalR))*(s - np.square(s))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * intervalR)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)

        u_30 = lambda x, y, z, s, t: (x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(intervalL - np.square(intervalL))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_31 = lambda x, y, z, s, t: (x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(intervalR - np.square(intervalR))*(t - np.square(t)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * intervalR)* np.sin(5*np.pi * t)

        u_40 = lambda x, y, z, s, t: (x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(s - np.square(intervalL))*(t - np.square(intervalL)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * intervalL) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * t)
        u_41 = lambda x, y, z, s, t:(x - np.square(x))*(y - np.square(y))*(z - np.square(z))*(s - np.square(s))*(intervalR - np.square(intervalR)) + \
                                       0.05*np.sin(5*np.pi * x) * np.sin(10*np.pi * y) * np.sin(20*np.pi * z)* np.sin(10*np.pi * s)* np.sin(5*np.pi * intervalR)
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41

    if equa_name == 'multi_scale5D_8':
        u_true = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))
        fside = lambda x, y, z, s, t: -(4*(np.pi*x)**2)*np.cos(2*np.pi*x)*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t)) \
                                      -(16*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*np.cos(4*np.pi*y)*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t)) \
                                      -(64*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*np.cos(8*np.pi*z)*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t)) \
                                      -(16*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*np.cos(4*np.pi*s)*(1-np.cos(2*np.pi*t)) \
                                      -(4*(np.pi*x)**2)*(1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*np.cos(2*np.pi*t)

        A_eps = lambda x, y, z, s, t: np.ones_like(x)

        u_00 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*intervalL))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))
        u_01 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*intervalR))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))

        u_10 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*intervalL))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))
        u_11 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*intervalR))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))

        u_20 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*intervalL))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))
        u_21 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*intervalR))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*t))

        u_30 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*intervalL))*(1-np.cos(2*np.pi*t))
        u_31 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*intervalR))*(1-np.cos(2*np.pi*t))

        u_40 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*intervalL))
        u_41 = lambda x, y, z, s, t: (1-np.cos(2*np.pi*x))*(1-np.cos(4*np.pi*y))*(1-np.cos(8*np.pi*z))*(1-np.cos(4*np.pi*s))*(1-np.cos(2*np.pi*intervalR))
        return u_true, fside, A_eps, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41


def get_forceSide2pLaplace5D(x=None, y=None, z=None, s=None, t=None, equa_name='multi_scale5D_4'):
    if equa_name == 'multi_scale5D_4':
        # u = np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        Aeps = 0.25*(1.0+np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t))

        ux = 1.0*np.pi*np.cos(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        uy = 10*np.pi*np.sin(np.pi*x)*np.cos(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        uz = 20*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.cos(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        us = 10*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.cos(10*np.pi*s)*np.sin(np.pi*t)
        ut = 1.0*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.cos(np.pi*t)

        uxx = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        uyy = -100*np.pi*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        uzz = -400*np.pi*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        uss = -100*np.pi*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)
        utt = -1.0*np.pi*np.pi*np.sin(np.pi*x)*np.sin(10*np.pi*y)*np.sin(20*np.pi*z)*np.sin(10*np.pi*s)*np.sin(np.pi*t)

        Aepsx = -0.25*np.pi*np.sin(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)
        Aepsy = -0.25*10*np.pi*np.cos(np.pi*x)*np.sin(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)
        Aepsz = -0.25*20*np.pi*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.sin(20*np.pi*z)*np.cos(10*np.pi*s)*np.cos(np.pi*t)
        Aepss = -0.25*10*np.pi*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.sin(10*np.pi*s)*np.cos(np.pi*t)
        Aepst = -0.25*np.pi*np.cos(np.pi*x)*np.cos(10*np.pi*y)*np.cos(20*np.pi*z)*np.cos(10*np.pi*s)*np.sin(np.pi*t)

        fside = -1.0*(Aepsx*ux + Aepsy*uy + Aepsz*uz + Aepss*us + Aepst*ut) - 1.0*Aeps*(uxx+uyy+uzz+uss+utt)
        return fside


def get_infos2pLaplace_10D(input_dim=1, out_dim=1, mesh_number=2, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'multi_scale10D_1':
        fside = lambda x, y, z: 10.0 * ((np.pi)**2) * (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))
        u_true = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        A_eps = lambda x, y, z: 1.0
        return u_true, fside, A_eps