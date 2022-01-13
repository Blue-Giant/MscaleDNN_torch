import numpy as np


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  方形区域[a,b]^n生成随机数, n代表变量个数
def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


def rand_bd_1D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 1:
        x_left_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_a
        x_right_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_b
        return x_left_bd, x_right_bd
    else:
        return


def rand_bd_2D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 2:
        x_left_bd = (region_b-region_a) * np.random.random([batch_size, 2]) + region_a   # 浮点数都是从0-1中随机。
        for ii in range(batch_size):
            x_left_bd[ii, 0] = region_a

        x_right_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            x_right_bd[ii, 0] = region_b

        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_bottom_bd[ii, 1] = region_a

        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_top_bd[ii, 1] = region_b

        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
        return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd
    else:
        return


def rand_bd_3D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 3:
        bottom_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            bottom_bd[ii, 2] = region_a

        top_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            top_bd[ii, 2] = region_b

        left_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            left_bd[ii, 1] = region_a

        right_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            right_bd[ii, 1] = region_b

        front_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            front_bd[ii, 0] = region_b

        behind_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            behind_bd[ii, 0] = region_a

        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)
        return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd
    else:
        return


def rand_bd_4D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    variable_dim = int(variable_dim)

    x0a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0a[ii, 0] = region_a

    x0b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0b[ii, 0] = region_b

    x1a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1a[ii, 1] = region_a

    x1b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1b[ii, 1] = region_b

    x2a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2a[ii, 2] = region_a

    x2b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2b[ii, 2] = region_b

    x3a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3a[ii, 3] = region_a

    x3b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3b[ii, 3] = region_b

    x0a = x0a.astype(np.float32)
    x0b = x0b.astype(np.float32)

    x1a = x1a.astype(np.float32)
    x1b = x1b.astype(np.float32)

    x2a = x2a.astype(np.float32)
    x2b = x2b.astype(np.float32)

    x3a = x3a.astype(np.float32)
    x3b = x3b.astype(np.float32)

    return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b


def rand_bd_5D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 5:
        x0a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0a[ii, 0] = region_a

        x0b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0b[ii, 0] = region_b

        x1a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1a[ii, 1] = region_a

        x1b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1b[ii, 1] = region_b

        x2a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2a[ii, 2] = region_a

        x2b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2b[ii, 2] = region_b

        x3a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3a[ii, 3] = region_a

        x3b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3b[ii, 3] = region_b

        x4a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4a[ii, 4] = region_a

        x4b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4b[ii, 4] = region_b

        x0a = x0a.astype(np.float32)
        x0b = x0b.astype(np.float32)

        x1a = x1a.astype(np.float32)
        x1b = x1b.astype(np.float32)

        x2a = x2a.astype(np.float32)
        x2b = x2b.astype(np.float32)

        x3a = x3a.astype(np.float32)
        x3b = x3b.astype(np.float32)

        x4a = x4a.astype(np.float32)
        x4b = x4b.astype(np.float32)
        return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, x4a, x4b
    else:
        return
