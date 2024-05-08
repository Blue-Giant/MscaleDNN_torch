# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io
import torch


# load the data from matlab of .mat
def load_Matlab_data(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_data2pLaplace(equation_name=None, mesh_number=2, to_torch=False, to_float=True, to_cuda=False, gpu_no=0,
                      use_grad2x=False):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = '../dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = '../dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = '../dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = '../dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = '../dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        test_meshXY_file = '../dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = '../dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = load_Matlab_data(test_meshXY_file)
    XY_points = mesh_XY['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        test_xy_data = np.transpose(XY_points, (1, 0))
    else:
        test_xy_data = XY_points

    if to_float:
        test_xy_data = test_xy_data.astype(np.float32)

    if to_torch:
        test_xy_data = torch.from_numpy(test_xy_data)

        if to_cuda:
            test_xy_data = test_xy_data.cuda(device='cuda:' + str(gpu_no))

        test_xy_data.requires_grad = use_grad2x
    return test_xy_data


def get_RandData2Mat(dim=2, data_path=None, to_torch=True, to_float=True, to_cuda=False, gpu_no=0, use_grad=False):
    if dim == 2:
        testData_file = str(data_path) + '/' + str('testXY') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XY']
        szie2test_data = int(np.size(data2test, axis=-1))
        if szie2test_data != dim:
            data2test = np.transpose(data2test, (1, 0))
    elif dim == 3:
        testData_file = str(data_path) + '/' + str('testXYZ') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZ']
        szie2test_data = int(np.size(data2test, axis=-1))
        if szie2test_data != dim:
            data2test = np.transpose(data2test, (1, 0))
    elif dim == 4:
        testData_file = str(data_path) + '/' + str('testXYZS') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZS']
        szie2test_data = int(np.size(data2test, axis=-1))
        if szie2test_data != dim:
            data2test = np.transpose(data2test, (1, 0))
    elif dim == 5:
        testData_file = str(data_path) + '/' + str('testXYZST') + str('.mat')
        testData = loadMatlabIdata(testData_file)
        data2test = testData['XYZST']
        szie2test_data = int(np.size(data2test, axis=-1))
        if szie2test_data != dim:
            data2test = np.transpose(data2test, (1, 0))

    if to_float:
        data2test = data2test.astype(np.float32)

    if to_torch:
        data2test = torch.from_numpy(data2test)

        if to_cuda:
            data2test = data2test.cuda(device='cuda:' + str(gpu_no))

        data2test.requires_grad = use_grad

    return data2test


# Load uniform mesh points for the inner of a 2D regular square(or rectangle) domain
def load_MatData2Mesh_2D(path2file=None, num2mesh=2, to_float=True, float_type=np.float32, to_torch=False,
                         to_cuda=False, gpu_no=0, use_grad=False, if_scaleXmesh=False, scaleX=10.0, Xbase=0.0,
                         if_scaleYmesh=False, scaleY=3.0, Ybase=0.0):
    test_meshXY_file = path2file + str('meshXY') + str(num2mesh) + str('.mat')
    mesh_points = loadMatlabIdata(test_meshXY_file)
    mesh_XY = mesh_points['meshXY']
    shape2XY = np.shape(mesh_XY)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_mesh = np.transpose(mesh_XY, (1, 0))
    else:
        xy_mesh = mesh_XY

    if to_float:
        xy_mesh = xy_mesh.astype(dtype=float_type)

    if if_scaleXmesh is True:
        xy_mesh[:, 0:1] = Xbase + xy_mesh[:, 0:1] * scaleX

    if if_scaleYmesh is True:
        xy_mesh[:, 1:2] = Ybase + xy_mesh[:, 1:2] * scaleY

    if to_torch:
        xy_mesh = torch.from_numpy(xy_mesh)

        if to_cuda:
            xy_mesh = xy_mesh.cuda(device='cuda:' + str(gpu_no))

        xy_mesh.requires_grad = use_grad

    return xy_mesh


# Load random points for the inner of a 2D irregular domain
def load_MatData2IrregularDomain_2D(path2file=None, to_float=True, float_type=np.float32, to_torch=False,
                                    to_cuda=False, gpu_no=0, use_grad=False, if_scaleXmesh=False, scaleX=10.0,
                                    Xbase=0.0, if_scaleYmesh=False, scaleY=3.0, Ybase=0.0):
    testXY_file = path2file + str('testXY') + str('.mat')
    Mat_XY_points = loadMatlabIdata(testXY_file)
    XY_Points = Mat_XY_points['meshXY']
    shape2XY = np.shape(XY_Points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        XY_Points = np.transpose(XY_Points, (1, 0))

    if to_float:
        XY_Points = XY_Points.astype(dtype=float_type)

    if if_scaleXmesh is True:
        XY_Points[:, 0:1] = Xbase + XY_Points[:, 0:1] * scaleX

    if if_scaleYmesh is True:
        XY_Points[:, 1:2] = Ybase + XY_Points[:, 1:2] * scaleY

    if to_torch:
        XY_Points = torch.from_numpy(XY_Points)

        if to_cuda:
            XY_Points = XY_Points.cuda(device='cuda:' + str(gpu_no))

        XY_Points.requires_grad = use_grad

    return XY_Points


# Load random points for the inner of a 2D porous domain
def load_InnerData2Porous_Domain_2D(path2file=None, num2index=5, region_left=0.0, region_right=0.0, region_bottom=0.0,
                                    region_top=0.0, to_float=True, float_type=np.float32, to_torch=False, to_cuda=False,
                                    gpu_no=0, use_grad=False, if_scaleX=False, scaleX=10.0, Xbase=0.0, if_scaleY=False,
                                    scaleY=3.0, Ybase=0.0):
    data_path = path2file + str('xy_porous') + str(num2index) + str('.txt')
    # data_path = '../data2PorousDomain_2D/Normalized/xy_porous5.txt'
    porous_points2xy = np.loadtxt(data_path)
    if to_float:
        porous_points2xy = porous_points2xy.astype(dtype=float_type)
    shape2data = np.shape(porous_points2xy)
    num2points = shape2data[0]
    points = []
    for ip in range(num2points):
        point = np.reshape(porous_points2xy[ip], newshape=(-1, 2))
        if point[0, 0] == region_left or point[0, 0] == region_right:
            continue
        elif point[0, 1] == region_bottom or point[0, 1] == region_top:
            continue
        else:
            points.append(point)
    xy_inside = np.concatenate(points, axis=0)

    if if_scaleX is True:
        xy_inside[:, 0:1] = scaleX * xy_inside[:, 0:1] + Xbase

    if if_scaleY is True:
        xy_inside[:, 1:2] = scaleY * xy_inside[:, 1:2] + Ybase

    if to_torch:
        xy_mesh = torch.from_numpy(xy_inside)

        if to_cuda:
            xy_mesh = xy_mesh.cuda(device='cuda:' + str(gpu_no))

        xy_mesh.requires_grad = use_grad

    return xy_inside


def load_FullData2Porous_Domain_2D(path2file=None, num2index=5, region_left=0.0, region_right=0.0, region_bottom=0.0,
                                    region_top=0.0, to_float=True, float_type=np.float32, to_torch=False, to_cuda=False,
                                    gpu_no=0, use_grad=False, if_scaleX=False, scaleX=10.0, Xbase=0.0, if_scaleY=False,
                                    scaleY=3.0, Ybase=0.0):
    data_path = path2file + str('xy_porous') + str(num2index) + str('.txt')
    # data_path = '../data2PorousDomain_2D/Normalized/xy_porous5.txt'
    porous_points2xy = np.loadtxt(data_path)
    if to_float:
        porous_points2xy = porous_points2xy.astype(dtype=float_type)
    shape2data = np.shape(porous_points2xy)
    num2points = shape2data[0]
    points = []
    for ip in range(num2points):
        point = np.reshape(porous_points2xy[ip], newshape=(-1, 2))
        points.append(point)
    xy_inside = np.concatenate(points, axis=0)

    if if_scaleX is True:
        xy_inside[:, 0:1] = scaleX * xy_inside[:, 0:1] + Xbase

    if if_scaleY is True:
        xy_inside[:, 1:2] = scaleY * xy_inside[:, 1:2] + Ybase

    if to_torch:
        xy_inside = torch.from_numpy(xy_inside)

        if to_cuda:
            xy_inside = xy_inside.cuda(device='cuda:' + str(gpu_no))

        xy_inside.requires_grad = use_grad

    return xy_inside


def get_MatData2Regular_Domain3D(data_path=None, to_torch=False, to_float=True,
                                 to_cuda=False, gpu_no=0, use_grad=False):
    file_name2data = str(data_path) + '/' + str('testXYZ') + str('.mat')
    data2matlab = loadMatlabIdata(filename=file_name2data)
    data2points = data2matlab['XYZ']
    shape2data = np.shape(data2points)
    assert (len(shape2data) == 2)
    if shape2data[0] == 3:
        data2points = np.transpose(data2points, (1, 0))

    if to_float:
        data2points = data2points.astype(np.float32)

    if to_torch:
        data2points = torch.from_numpy(data2points)

        if to_cuda:
            data2points = data2points.cuda(device='cuda:' + str(gpu_no))

        data2points.requires_grad = use_grad
    return data2points


def get_MatData2Holes_3D(data_path=None, to_float=True, to_torch=False, to_cuda=False, gpu_no=0, use_grad=False):
    file_name2data = str(data_path) + '/' + str('TwoSlice2TestXYZ_01') + str('.mat')
    rand_points = loadMatlabIdata(file_name2data)
    XYZ_points = rand_points['XYZ']
    shape2XYZ = np.shape(XYZ_points)
    assert(len(shape2XYZ) == 2)
    if shape2XYZ[0] == 3:
        xyz_data = np.transpose(XYZ_points, (1, 0))
    else:
        xyz_data = XYZ_points

    if to_float:
        xyz_data = xyz_data.astype(np.float32)

    if to_torch:
        xyz_data = torch.from_numpy(xyz_data)

        if to_cuda:
            xyz_data = xyz_data.cuda(device='cuda:' + str(gpu_no))

        xyz_data.requires_grad = use_grad
    return xyz_data


if __name__ == '__main__':
    mat_data_path = '../dataMat_highDim'
    mat_data = get_RandData2Mat(dim=2, data_path=mat_data_path)
    print('end!!!!')