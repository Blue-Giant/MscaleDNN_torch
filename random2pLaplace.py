import torch as th
import numpy as np


def random_boundary():
    ul = lambda x: 0.0
    ur = lambda x: 0.0
    return ul, ur


def rangdom_ceof1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = th.pow(float(k+1), -1.0*alpha)*(xi1k*th.sin(k*x)+xi2k*th.cos(k*x))
        sum = sum + temp
    a = a+0.5*th.sin(sum)
    return a


def rangdom_force1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = th.pow(float(k+1), -1.0*alpha)*(xi1k*th.sin(k*x)+xi2k*th.cos(k*x))
        sum = sum + temp
    return sum


def rangdom_diff_force2x_1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = th.pow(float(k+1), -1.0*alpha)*(k*xi1k*th.cos(k*x)-k*xi2k*th.sin(k*x))
        sum = sum + temp
    return sum


def rangdom_exact_solution_1(x=None, xi1=None, xi2=None, K=5, alpha=0.8):
    a = 1.0
    sum = 0.0
    for k in range(K):
        xi1k = xi1[k]
        xi2k = xi2[k]
        temp = th.pow(float(k+1), -1.0*alpha)*(xi1k*th.sin(k*x)+xi2k*th.cos(k*x))
        sum = sum + temp
    a = a+0.5*th.sin(sum)
    u = th.div(sum, a)
    return u


def random_equa2(xi1=0.25, xi2=-0.25, K=2, alpha=1.0):
    u = lambda x: th.sin(xi1*th.sin(np.pi*x)+0.5*xi2*th.sin(2*np.pi*x))
    dux = lambda x: th.cos(xi1*th.sin(np.pi*x)+th.pow(2, -1.0*alpha)*xi2*th.sin(2*np.pi*x))*0.25*np.pi*(th.cos(2*np.pi*x)-th.cos(np.pi*x))
    f = lambda x: 1.0
    aeps = lambda x: 1.0
    return u, f, aeps


def random_sin_f(x=None, xi1=0.25, xi2=-0.25, K=2, alpha=1.0):
    f = th.sin(xi1*th.sin(np.pi*x)+0.5*xi2*th.sin(2*np.pi*x))*np.pi*(xi2*th.cos(2*np.pi*x)+xi1*th.cos(np.pi*x))* \
        np.pi * (xi2*th.cos(2*np.pi*x) + xi1*th.cos(np.pi*x)) - \
        th.cos(xi1*th.sin(np.pi*x) + 0.5*xi2*th.sin(2*np.pi*x)) * np.pi * \
        (-1.0*np.pi*xi1*th.sin(np.pi*x) - 1.0*2.0* np.pi *xi2* th.sin(2*np.pi*x))
    return f
