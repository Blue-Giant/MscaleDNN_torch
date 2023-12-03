import torch


def compute_DNN_1st_grads_1D(UNN=None, input_variable=None, grad_shape2out=None):
    assert UNN is not None
    assert input_variable is not None
    assert grad_shape2out is not None
    grad2UNN = torch.autograd.grad(UNN, input_variable, grad_outputs=torch.ones_like(UNN),
                                   create_graph=True, retain_graph=True)
    dUNN2dx = grad2UNN[0]
    return dUNN2dx


def compute_DNN_1st_grads_2D(UNN=None, input_variable=None, grad_shape2out=None):
    assert UNN is not None
    assert input_variable is not None
    assert grad_shape2out is not None
    grad2UNN = torch.autograd.grad(UNN, input_variable, grad_outputs=torch.ones_like(UNN),
                                   create_graph=True, retain_graph=True)
    dUNN2dxy = grad2UNN[0]

    dUNN2dx = torch.reshape(dUNN2dxy[:, 0], shape=[-1, 1])
    dUNN2dy = torch.reshape(dUNN2dxy[:, 1], shape=[-1, 1])
    return dUNN2dx, dUNN2dy


def compute_DNN_1st_grads_3D(UNN=None, input_variable=None, grad_shape2out=None):
    assert UNN is not None
    assert input_variable is not None
    assert grad_shape2out is not None
    grad2UNN = torch.autograd.grad(UNN, input_variable, grad_outputs=torch.ones_like(UNN),
                                   create_graph=True, retain_graph=True)
    dUNN2dxyz = grad2UNN[0]

    dUNN2dx = torch.reshape(dUNN2dxyz[:, 0], shape=[-1, 1])
    dUNN2dy = torch.reshape(dUNN2dxyz[:, 1], shape=[-1, 1])
    dUNN2dz = torch.reshape(dUNN2dxyz[:, 2], shape=[-1, 1])
    return dUNN2dx, dUNN2dy, dUNN2dz


def compute_DNN_1st_grads_4D(UNN=None, input_variable=None, grad_shape2out=None):
    assert UNN is not None
    assert input_variable is not None
    assert grad_shape2out is not None
    grad2UNN = torch.autograd.grad(UNN, input_variable, grad_outputs=torch.ones_like(UNN),
                                   create_graph=True, retain_graph=True)
    dUNN2dxyzs = grad2UNN[0]

    dUNN2dx = torch.reshape(dUNN2dxyzs[:, 0], shape=[-1, 1])
    dUNN2dy = torch.reshape(dUNN2dxyzs[:, 1], shape=[-1, 1])
    dUNN2dz = torch.reshape(dUNN2dxyzs[:, 2], shape=[-1, 1])
    dUNN2ds = torch.reshape(dUNN2dxyzs[:, 3], shape=[-1, 1])
    return dUNN2dx, dUNN2dy, dUNN2dz, dUNN2ds


def compute_DNN_1st_grads_5D(UNN=None, input_variable=None, grad_shape2out=None):
    assert UNN is not None
    assert input_variable is not None
    assert grad_shape2out is not None
    grad2UNN = torch.autograd.grad(UNN, input_variable, grad_outputs=torch.ones_like(UNN),
                                   create_graph=True, retain_graph=True)
    dUNN2dxyzst = grad2UNN[0]

    dUNN2dx = torch.reshape(dUNN2dxyzst[:, 0], shape=[-1, 1])
    dUNN2dy = torch.reshape(dUNN2dxyzst[:, 1], shape=[-1, 1])
    dUNN2dz = torch.reshape(dUNN2dxyzst[:, 2], shape=[-1, 1])
    dUNN2ds = torch.reshape(dUNN2dxyzst[:, 3], shape=[-1, 1])
    dUNN2dt = torch.reshape(dUNN2dxyzst[:, 3], shape=[-1, 1])
    return dUNN2dx, dUNN2dy, dUNN2dz, dUNN2ds, dUNN2dt
