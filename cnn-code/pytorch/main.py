import cnn_operator as cnn_opt
import torch
import torchvision as tv
import torch.nn as nn

import math
import numpy as np
import os,sys
if __name__ == '__main__':

# #CONV operator
#     conv_fm_in_shape   = [1, 11 ,176, 176]     #输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
#     conv_kernel_shape  = [10, 11, 5, 5]      #卷积核参数 ====shape为 [ out_channels,in_channel, filter_height, filter_weight]==
#     conv_stride_info   = [1, 1, 7, 7]      #步进参数   ====shape为 [ 1, 1, row_strides, col_strides]
#     conv_dilation_info = [1, 1, 3, 3]
#     conv_padding_info  = [13,13]
#     # cnn_opt.cnn_operator_conv(conv_fm_in_shape,conv_kernel_shape,conv_stride_info,conv_dilation_info,conv_padding_info)
#
#     act_fm_in_shape    = [1, 3, 256, 256, ]  # 输入特征图 ====shape为 [ batch, in_channel, in_height, in_weight]===
#     act_prelu_para_init= np.float32(0.28)
#     # act_prelu_para_init= torch.double(0.28) #.float64(0.28)
#     act_layer_leakyrelu_para= np.float32(1e-2)
#     fm_in = (torch.randn(tuple(act_fm_in_shape))*6)
#
#     act_layer_relu      = nn.ReLU(inplace=False)
#     act_layer_prelu     = nn.PReLU(act_fm_in_shape[1], act_prelu_para_init)                # a参数可学习 max(0,x) + a * min(0,x)
#     act_layer_leakyrelu = nn.LeakyReLU(act_layer_leakyrelu_para, inplace=False)
#     act_layer_relu6     = nn.ReLU6(inplace=False)
#     act_layer_sigmoid   = nn.Sigmoid()                                           # 1 / ( 1 + e^{-x}
#     act_layer_Tanh      = nn.Tanh()
#
#     fm_out_relu      = act_layer_relu.forward(fm_in)
#     fm_out_prelu     = act_layer_prelu.forward(fm_in)
#     fm_out_leakyrelu = act_layer_leakyrelu.forward(fm_in)
#     fm_out_relu6     = act_layer_relu6.forward(fm_in)
#     fm_out_sigmoid   = act_layer_sigmoid.forward(fm_in)
#     fm_out_Tanh      = act_layer_Tanh.forward(fm_in)
#
#     dir = 'inout'
#     if not os.path.exists(dir):
#         os.mkdir(dir)
#
#     np.savetxt(dir + '/act_fm_in.txt'            , fm_in.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_relu.txt'      , fm_out_relu.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_prelu.txt'     , fm_out_prelu.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_leakyrelu.txt' , fm_out_leakyrelu.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_relu6.txt'     , fm_out_relu6.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_sigmoid.txt'   , fm_out_sigmoid.detach().numpy().reshape(-1, 1))
#     np.savetxt(dir + '/act_fm_out_Tanh.txt'      , fm_out_Tanh.detach().numpy().reshape(-1, 1))
#
#     print("==================================================")
#     print("=the ACT   process done DB have saved  in the file=")
#     print("==================================================")











#POOL operator
    pool_mode = "AVG"
    pool_fm_in_shape    = [1, 3, 256, 256, ]  # 输入特征图 ====shape为 [ batch, in_channel, in_height, in_weight]===
    pool_kernel_info    = [1, 1, 2, 2]
    pool_stride_info    = [1, 1, 7, 7]
    pool_dilation_info  = [1, 1, 3, 3]
    pool_padding_info   = [1, 1]
    # cnn_opt.cnn_operator_pool(pool_fm_in_shape,pool_kernel_info,pool_stride_info,pool_dilation_info,pool_padding_info,pool_mode)









# FC operator
    batch_size = 10
    fm_in_node_num = 20
    fm_out_node_num = 30
    # cnn_opt.cnn_operator_fc(batch_size, fm_in_node_num, fm_out_node_num)

# Batch-Normlize operator
#     fm_in = np.array(np.arange(1,101,1),dtype='d')

    BN_fm_in_shape = [10,3, 64, 64]
    num_feature = BN_fm_in_shape[1]
    fm_in = np.array(np.random.randn(BN_fm_in_shape[0],BN_fm_in_shape[1],BN_fm_in_shape[2],BN_fm_in_shape[3]), dtype='d')
    # fm_in_tmp = fm_in.reshape((BN_fm_in_shape[0],BN_fm_in_shape[1],BN_fm_in_shape[2],BN_fm_in_shape[3]))
    BN_fm_in = torch.from_numpy(fm_in)
    # BN_fm_in_shape    = [16, 3, 32, 32]   #输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===

    ##== Without Learnable Parameters
    gamma = np.float64(1)
    beta  = np.float64(0)
    eps   = np.float64(0)
    momentum = np.float64(1)
    BN_layer = nn.BatchNorm2d(num_feature,eps=eps,momentum=momentum,affine=False)
    BN_layer.running_mean.data=BN_layer.running_mean.data.to(torch.float64)     #torch dataformat change
    BN_layer.running_var.data =BN_layer.running_var.data.to(torch.float64)
    BN_fm_out= BN_layer.forward(BN_fm_in)

    for i in range(0,BN_layer.running_mean.data.shape[0]):
        print('batch_mean: {:.10} batch var: {:.20f}'.format(BN_layer.running_mean.data[i], BN_layer.running_var.data[i]))
#torch2numpy
    fm_in_np  = BN_fm_in.detach().numpy()
    fm_out_np = BN_fm_out.detach().numpy()

    mean_0 = BN_layer.running_mean.data[0]
    var_0 = BN_layer.running_var.data[0]
    fm_out_0000=(fm_in_np[0,0,0,0]-mean_0)/(np.sqrt(var_0)+eps)*gamma+beta
    print('fm_out0000 var: {:.20f}'.format(fm_out_0000))
#文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

    np.savetxt(dir + '/BN_fm_in.txt' , BN_fm_in.reshape(-1, 1))
    np.savetxt(dir + '/BN_fm_out.txt', BN_fm_out.reshape(-1, 1))

    print("==================================================")
    print("=the BN   process done DB have saved  in the file=")
    print("==================================================")
