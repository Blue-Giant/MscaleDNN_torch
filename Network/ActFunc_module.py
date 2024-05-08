import torch
import torch.nn as tn
import torch.nn.functional as tnf

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
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*torch.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*torch.sin(x_input) + 0.5*torch.cos(x_input)
            # out_x = 0.75*torch.sin(x_input) + 0.75*torch.cos(x_input)
            # out_x = torch.sin(x_input) + torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        elif str.lower(self.actName) == 'gelu':
            out_x = tnf.gelu(x_input)
        elif str.lower(self.actName) == 'gcu':
            out_x = x_input*torch.cos(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = tnf.mish(x_input)
        elif str.lower(self.actName) == 'hard_swish':
            out_x = tnf.hardswish(x_input)
        elif str.lower(self.actName) == 'silu':
            out_x = tnf.silu(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = torch.exp(-1.0 * x_input * x_input)
        elif str.lower(self.actName) == 'requ':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'recu':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'morlet':
            out_x = torch.cos(1.75*x_input)*torch.exp(-0.5*x_input*x_input)
        else:
            out_x = x_input
        return out_x
