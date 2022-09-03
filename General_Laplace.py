import numpy as np
import torch


def get_infos2Laplace_1D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    # -uxx = f
    if equa_name == 'PDE1':
        # u=sin(pi*x), f=-pi*pi*sin(pi*x)
        fside = lambda x: -(torch.pi)*(torch.pi)*torch.sin(torch.pi*x)
        utrue = lambda x: torch.sin(torch.pi*x)
        uleft = lambda x: torch.sin(torch.pi*x)
        uright = lambda x: torch.sin(torch.pi*x)
    return fside, utrue, uleft, uright


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_2D(input_dim=1, out_dim=1, left_bottom=0.0, right_top=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # u=exp(-x)(x_y^3), f = -exp(-x)(x-2+y^3+6y)
        f_side = lambda x, y: -(np.exp(-1.0*x)) * (x - 2 + np.power(y, 3) + 6 * y)

        u_true = lambda x, y: (np.exp(-1.0*x))*(x + np.power(y, 3))

        ux_left = lambda x, y: np.exp(-left_bottom) * (np.power(y, 3) + 1.0 * left_bottom)
        ux_right = lambda x, y: np.exp(-right_top) * (np.power(y, 3) + 1.0 * right_top)
        uy_bottom = lambda x, y: np.exp(-x) * (np.power(left_bottom, 3) + x)
        uy_top = lambda x, y: np.exp(-x) * (np.power(right_top, 3) + x)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE2':
        f_side = lambda x, y: (-1.0)*np.sin(np.pi*x) * (2 - np.square(np.pi)*np.square(y))

        u_true = lambda x, y: np.square(y)*np.sin(np.pi*x)

        ux_left = lambda x, y: np.square(y) * np.sin(np.pi * left_bottom)
        ux_right = lambda x, y: np.square(y) * np.sin(np.pi * right_top)
        uy_bottom = lambda x, y: np.square(left_bottom) * np.sin(np.pi * x)
        uy_top = lambda x, y: np.square(right_top) * np.sin(np.pi * x)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE3':
        # u=exp(x+y), f = -2*exp(x+y)
        f_side = lambda x, y: -2.0*(np.exp(x)*np.exp(y))
        u_true = lambda x, y: np.exp(x)*np.exp(y)
        ux_left = lambda x, y: np.multiply(np.exp(y), np.exp(left_bottom))
        ux_right = lambda x, y: np.multiply(np.exp(y), np.exp(right_top))
        uy_bottom = lambda x, y: np.multiply(np.exp(x), np.exp(left_bottom))
        uy_top = lambda x, y: np.multiply(np.exp(x), np.exp(right_top))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE4':
        # u=(1/4)*(x^2+y^2), f = -1
        f_side = lambda x, y: -1.0*np.ones_like(x)
        u_true = lambda x, y: 0.25*(np.pow(x, 2)+np.pow(y, 2))
        ux_left = lambda x, y: 0.25 * np.pow(y, 2) + 0.25 * np.pow(left_bottom, 2)
        ux_right = lambda x, y: 0.25 * np.pow(y, 2) + 0.25 * np.pow(right_top, 2)
        uy_bottom = lambda x, y: 0.25 * np.pow(x, 2) + 0.25 * np.pow(left_bottom, 2)
        uy_top = lambda x, y: 0.25 * np.pow(x, 2) + 0.25 * np.pow(right_top, 2)
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE5':
        # u=(1/4)*(x^2+y^2)+x+y, f = -1
        f_side = lambda x, y: -1.0*np.ones_like(x)

        u_true = lambda x, y: 0.25*(np.pow(x, 2)+np.pow(y, 2)) + x + y

        ux_left = lambda x, y: 0.25 * np.pow(y, 2) + 0.25 * np.pow(left_bottom, 2) + left_bottom + y
        ux_right = lambda x, y: 0.25 * np.pow(y, 2) + 0.25 * np.pow(right_top, 2) + right_top + y
        uy_bottom = lambda x, y: 0.25 * np.pow(x, 2) + np.pow(left_bottom, 2) + left_bottom + x
        uy_top = lambda x, y: 0.25 * np.pow(x, 2) + 0.25 * np.pow(right_top, 2) + right_top + x
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE6':
        # u=(1/2)*(x^2)*(y^2), f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(np.pow(x, 2)+np.pow(y, 2))

        u_true = lambda x, y: 0.5 * (np.pow(x, 2) * np.pow(y, 2))

        ux_left = lambda x, y: 0.5 * (np.pow(left_bottom, 2) * np.pow(y, 2))
        ux_right = lambda x, y: 0.5 * (np.pow(right_top, 2) * np.pow(y, 2))
        uy_bottom = lambda x, y: 0.5 * (np.pow(x, 2) * np.pow(left_bottom, 2))
        uy_top = lambda x, y: 0.5 * (np.pow(x, 2) * np.pow(right_top, 2))
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top
    elif equa_name == 'PDE7':
        # u=(1/2)*(x^2)*(y^2)+x+y, f = -(x^2+y^2)
        f_side = lambda x, y: -1.0*(np.pow(x, 2)+np.pow(y, 2))

        u_true = lambda x, y: 0.5*(np.pow(x, 2)*np.pow(y, 2)) + x*np.ones_like(x) + y*np.ones_like(y)

        ux_left = lambda x, y: 0.5 * np.multiply(np.pow(left_bottom, 2), np.pow(y, 2)) + left_bottom + y
        ux_right = lambda x, y: 0.5 * np.multiply(np.pow(right_top, 2), np.pow(y, 2)) + right_top + y
        uy_bottom = lambda x, y: 0.5 * np.multiply(np.pow(x, 2), np.pow(left_bottom, 2)) + x + left_bottom
        uy_top = lambda x, y: 0.5 * np.multiply(np.pow(x, 2), np.pow(right_top, 2)) + x + right_top
        return f_side, u_true, ux_left, ux_right, uy_bottom, uy_top


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_3D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # -Laplace U = f
        # u=sin(pi*x)*sin(pi*y)*sin(pi*z), f=-pi*pi*sin(pi*x)*sin(pi*y)*sin(pi*z)
        fside = lambda x, y, z: -(np.pi)*(np.pi)*np.sin(np.pi*x)
        utrue = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_00 = lambda x, y, z: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_01 = lambda x, y, z: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)
        u_10 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalL)*np.sin(np.pi*z)
        u_11 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*intervalR)*np.sin(np.pi*z)
        u_20 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalL)
        u_21 = lambda x, y, z: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*intervalR)
    return fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21


# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数
def get_infos2Laplace_5D(input_dim=1, out_dim=1, intervalL=0.0, intervalR=1.0, equa_name=None):
    if equa_name == 'PDE1':
        # u=sin(pi*x), f=-pi*pi*sin(pi*x)
        fside = lambda x, y, z, s, t: -(np.pi)*(np.pi)*np.sin(np.pi*x)
        utrue = lambda x, y, z, s, t: np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)
        u_00 = lambda x, y, z, s, t: np.sin(np.pi*intervalL)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)
        u_01 = lambda x, y, z, s, t: np.sin(np.pi*intervalR)*np.sin(np.pi*y)*np.sin(np.pi*z)*np.sin(np.pi*s)*np.sin(np.pi*t)
        u_10 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalL) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_11 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * intervalR) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_20 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalL) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_21 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * intervalR) * np.sin(np.pi * s) * np.sin(np.pi * t)
        u_30 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalL) * np.sin(np.pi * t)
        u_31 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * intervalR) * np.sin(np.pi * t)
        u_40 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalL)
        u_41 = lambda x, y, z, s, t: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z) * np.sin(np.pi * s) * np.sin(np.pi * intervalR)
    return fside, utrue, u_00, u_01, u_10, u_11, u_20, u_21, u_30, u_31, u_40, u_41