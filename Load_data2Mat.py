# !python3
# -*- coding: utf-8 -*-
# author: flag

import numpy as np
import scipy.io


# load the data from matlab of .mat
def load_Matlab_data(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def get_randomData2mat(dim=2, data_path=None):
    if dim == 2:
        file_name2data = str(data_path) + '/' + str('testXY') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XY']
    elif dim == 3:
        file_name2data = str(data_path) + '/' + str('testXYZ') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZ']
    elif dim == 4:
        file_name2data = str(data_path) + '/' + str('testXYZS') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZS']
    elif dim == 5:
        file_name2data = str(data_path) + '/' + str('testXYZST') + str('.mat')
        data2matlab = load_Matlab_data(file_name2data)
        data2points = data2matlab['XYZST']
    return data2points


def get_meshData2Laplace(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        assert (mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(test_meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert(len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    return xy_data


def get_data2pLaplace(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        test_meshXY_file = 'dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        test_meshXY_file = 'dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        test_meshXY_file = 'dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        test_meshXY_file = 'dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        test_meshXY_file = 'dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        test_meshXY_file = 'dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_7':
        assert(mesh_number == 6)
        test_meshXY_file = 'dataMat2pLaplace/E7/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_XY = load_Matlab_data(test_meshXY_file)
    XY = mesh_XY['meshXY']
    test_xy_data = np.transpose(XY, (1, 0))
    return test_xy_data


def get_meshData2Boltzmann(equation_name=None, domain_lr='01', mesh_number=2):
    if domain_lr == '01':
        meshXY_file = 'dataMat2Boltz/meshData_01/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif domain_lr == '11':
        meshXY_file = 'dataMat2Boltz/meshData_11/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    return xy_data


def get_meshdata2Convection(equation_name=None, mesh_number=2):
    if equation_name == 'multi_scale2D_1':
        meshXY_file = 'dataMat2pLaplace/E1/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_2':
        meshXY_file = 'dataMat2pLaplace/E2/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_3':
        meshXY_file = 'dataMat2pLaplace/E3/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_4':
        meshXY_file = 'dataMat2pLaplace/E4/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_5':
        meshXY_file = 'dataMat2pLaplace/E5/' + str('meshXY') + str(mesh_number) + str('.mat')
    elif equation_name == 'multi_scale2D_6':
        meshXY_file = 'dataMat2pLaplace/E6/' + str('meshXY') + str(mesh_number) + str('.mat')
    mesh_points = load_Matlab_data(meshXY_file)
    XY_points = mesh_points['meshXY']
    shape2XY = np.shape(XY_points)
    assert (len(shape2XY) == 2)
    if shape2XY[0] == 2:
        xy_data = np.transpose(XY_points, (1, 0))
    return xy_data


if __name__ == '__main__':
    mat_data_path = 'dataMat_highDim'
    mat_data = get_randomData2mat(dim=2, data_path=mat_data_path)
    print('end!!!!')