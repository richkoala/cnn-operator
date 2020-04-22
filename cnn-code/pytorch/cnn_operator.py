import torch
import torchvision as tv
import torch.nn as nn

import math
import numpy as np
import os,sys


def cnn_operator_conv(fm_in_shape,kernel_shape,conv_stride_info,conv_dilation_info,conv_padding_info):

#理论计算获取的输出特征图大小
    fm_out_shape = [
        fm_in_shape[0]
        ,kernel_shape[0]
        ,math.floor((fm_in_shape[2]+2*conv_padding_info[0]-conv_dilation_info[2]*(kernel_shape[2]-1)-1)/conv_stride_info[2]+1)
        ,math.floor((fm_in_shape[3]+2*conv_padding_info[1]-conv_dilation_info[3]*(kernel_shape[3]-1)-1)/conv_stride_info[3]+1)
    ]

#特征数据和卷积核生成
    fm_in_np  = np.random.randn(fm_in_shape[0],fm_in_shape[1],fm_in_shape[2],fm_in_shape[3])
    weight_np = np.random.randn(kernel_shape[0],kernel_shape[1],kernel_shape[2],kernel_shape[3])
    bias_np   = np.random.randn(kernel_shape[0])

#卷积层
    # op = nn.conv2d(fm_in, kernel, strides=conv_stride_info, dilations=conv_dilation_info, padding=conv_padding_mode)
    conv_layer = nn.Conv2d(kernel_shape[0],kernel_shape[1], tuple(kernel_shape[2:4]),
                  stride=tuple(conv_stride_info[2:4]),
                  padding=tuple(conv_padding_info[0:2]),
                  dilation=tuple(conv_dilation_info[2:4])
                  )

    conv_layer.weight = torch.nn.Parameter(torch.from_numpy(weight_np))
    conv_layer.bias   = torch.nn.Parameter(torch.from_numpy(bias_np))

    fm_out=conv_layer.forward(torch.from_numpy(fm_in_np))
    fm_out_np = fm_out.detach().numpy()

    print("the tensorflow calc  fm_out's shape is"  + str(fm_out.shape))
    print("the theory     calc  fm_out's shape is " + str(fm_out_shape))


#文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

#输入输出数据打印
    # print(fm_in_np)
    # print(weight_np)
    # print(bias_np)
    # print(fm_out_np)

    np.savetxt(dir + '/conv_fm_in.txt' , fm_in_np.reshape(-1,1))
    np.savetxt(dir + '/conv_weight.txt', weight_np.reshape(-1,1))
    np.savetxt(dir + '/conv_bias.txt'  , bias_np.reshape(-1,1))
    np.savetxt(dir + '/conv_fm_out.txt', fm_out_np.reshape(-1,1))

    print("\n\n")
    print("==================================================")
    print("=the Conv process done DB have saved  in the file=")
    print("==================================================")
    print("\n\n")

    # 维度转化
    # filter_tranpose = filter_np.transpose((3,2,0,1))    #[ filter_height, filter_weight, in_channel, out_channels ]
    # fm_in_tranpose  = fm_in_np.transpose((0,3,1,2))     #[ batch, in_height, in_weight, in_channel ]===
    # fm_out_tranpose = fm_out.transpose((0,3,1,2))       #[ batch, in_height, in_weight, in_channel ]===
    # np.savetxt('fm_in_tranpose.txt' , fm_in_tranpose.reshape(-1,1) )
    # np.savetxt('filter_tranpose.txt', filter_tranpose.reshape(-1,1))
    # np.savetxt('fm_out_tranpose.txt', fm_out_tranpose.reshape(-1,1))


def cnn_operator_pool(fm_in_shape,pool_kernel_info,pool_stride_info,pool_dilation_info,pool_padding_info,pool_mode):
    # pool_mode = "AVG"
    # fm_in_shape        = [1, 3,   256, 256, ] # 输入特征图 ====shape为 [ batch, in_channel, in_height, in_weight]===
    # pool_kernel_info   = [1, 1,   2,  2  ]
    # pool_stride_info   = [1, 1,   3,  3  ]
    # pool_dilation_info = [1, 1,   1,  1  ]
    # pool_padding_info  = [1 ,1]

#理论计算获取的输出特征图大小
    if pool_mode == "MAX":
        fm_out_shape =[
            fm_in_shape[0]
            ,fm_in_shape[1]
            ,math.floor( (fm_in_shape[2]+2*pool_padding_info[0]-pool_dilation_info[2]*(pool_kernel_info[2]-1)-1)/pool_stride_info[2] + 1)
            ,math.floor( (fm_in_shape[3]+2*pool_padding_info[1]-pool_dilation_info[3]*(pool_kernel_info[3]-1)-1)/pool_stride_info[3] + 1)

        ]
    elif pool_mode == "AVG":
        fm_out_shape =[
            fm_in_shape[0]
            ,fm_in_shape[1]
            ,math.floor( (fm_in_shape[2]+2*pool_padding_info[0]-pool_kernel_info[2]-1)/pool_stride_info[2] + 1)
            ,math.floor( (fm_in_shape[3]+2*pool_padding_info[1]-pool_kernel_info[3]-1)/pool_stride_info[3] + 1)

        ]


#特征数据生成
    fm_in_np = np.random.randn(fm_in_shape[0], fm_in_shape[1], fm_in_shape[2], fm_in_shape[3]);
    fm_in = torch.from_numpy(fm_in_np)

#池化层
    if pool_mode == "MAX":
        pool_layer = nn.MaxPool2d(tuple(pool_kernel_info[2:4]),tuple(pool_stride_info[2:4]),tuple(pool_padding_info[0:2]),tuple(pool_dilation_info[2:4]),False,True)
    elif pool_mode == "AVG":
        pool_layer = nn.AvgPool2d(tuple(pool_kernel_info[2:4]),tuple(pool_stride_info[2:4]),tuple(pool_padding_info[0:2]),False,True)
    else:
        print("pooling mode error")
        sys.exit(0)


    fm_out=pool_layer.forward(torch.from_numpy(fm_in_np))
    fm_out_np = fm_out.detach().numpy()

    print("the tensorflow calc fm_out_pool's shape is " + str(fm_out.shape))
    print("the theory     calc fm_out_pool's shape is " + str(fm_out_shape))

# 文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

    np.savetxt(dir+'/pool_fm_in_' +pool_mode+'.txt', fm_in_np.reshape(-1, 1))
    np.savetxt(dir+'/pool_fm_out_'+pool_mode+'.txt', fm_out.reshape(-1, 1))

    print("==================================================")
    print("=the Pool process done DB have saved  in the file=")
    print("==================================================")

def cnn_operator_fc(batch_size, fm_in_node_num, fm_out_node_num):
# 特征数据 与权重偏置数据生成

    fm_in_shape  = [batch_size, fm_in_node_num]  # 输入特征图 ====shape为 [batch_size,FmIn_node]===
    fm_out_shape = [batch_size, fm_out_node_num]  # 卷积核参数 ====shape为[batch_size,FmOut_node]==

    fm_in_np = np.random.randn(fm_in_shape[0], fm_in_shape[1])
    weight_np = np.random.randn(fm_out_shape[1], fm_in_shape[1])  # Note: weight data order (out_features , in_features)
    bias_np = np.random.randn(fm_out_shape[1])  # Note: bias   data order (out_features)
    fc_weight = torch.from_numpy(weight_np)
    fc_bias = torch.from_numpy(bias_np)
    fm_in = torch.from_numpy(fm_in_np)

    # FC全连接层
    fc_layer = nn.Linear(fm_in_shape[1], fm_out_shape[1], bias=True)
    fc_layer.weight = torch.nn.Parameter(fc_weight)
    fc_layer.bias = torch.nn.Parameter(fc_bias)

    fm_out = fc_layer.forward(fm_in)
    fm_out_np = fm_out.detach().numpy()

#文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

#输入输出数据打印
    np.savetxt(dir + '/fc_fm_in.txt' , fm_in_np.reshape(-1,1))
    np.savetxt(dir + '/fc_weight.txt', weight_np.reshape(-1,1))
    np.savetxt(dir + '/fc_bias.txt'  , bias_np.reshape(-1,1))
    np.savetxt(dir + '/fc_fm_out.txt', fm_out_np.reshape(-1,1))

    print("==================================================")
    print("==the FC process done DB have saved  in the file==")
    print("==================================================")
