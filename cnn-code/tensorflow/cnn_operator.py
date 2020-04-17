import tensorflow as tf
import math
import numpy as np
import os,sys


def cnn_operator_conv(fm_in_shape,filter_shape,conv_stride_info,conv_dilation_info,conv_padding_mode = 'SAME'):
# def cnn_operator_conv():

    # fm_in_shape        = [1, 256, 256, 3]   #输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
    # filter_shape       = [5, 5, 3, 16]      #卷积核参数 ====shape为 [ filter_height, filter_weight, in_channel, out_channels ]==
    # conv_stride_info   = [1, 2, 2, 1]       #步进参数   ====shape为 [ 1, strides, strides, 1]，第一位和最后一位固定必须是1
    # conv_dilation_info = [1, 1, 1, 1]

#tensorflow 的补零方式，如果修改为其他的学习框架可以自动完成conv_padding的填充处理
    # conv_padding_mode = 'VALID'
    # conv_padding_mode = 'SAME'
    if conv_padding_mode == 'VALID':
        conv_padding = [0, 0]
    elif conv_padding_mode == 'SAME':
        conv_padding = [
             (filter_shape[0] + (filter_shape[0]-1) * (conv_dilation_info[1] - 1) - 1) / 2
            ,(filter_shape[1] + (filter_shape[1]-1) * (conv_dilation_info[2] - 1) - 1) / 2
        ]
    else:
        print("padding mode error")
        sys.exit(0)
    print("conv_padding mode is ", str(conv_padding))

#理论计算获取的输出特征图大小
    fm_out_shape =[
        1
        ,math.floor( (fm_in_shape[1]-(filter_shape[0]+(filter_shape[0]-1)*(conv_dilation_info[1]-1))+2*conv_padding[0])/conv_stride_info[1]) + 1
        ,math.floor( (fm_in_shape[2]-(filter_shape[1]+(filter_shape[1]-1)*(conv_dilation_info[2]-1))+2*conv_padding[1])/conv_stride_info[2]) + 1
        ,filter_shape[3]
    ]

#特征数据和卷积核生成
    fm_in_np  = np.random.randn(fm_in_shape[0],fm_in_shape[1],fm_in_shape[2],fm_in_shape[3]);
    filter_np = np.random.randn(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]);
    fm_in= tf.convert_to_tensor(fm_in_np)
    filter= tf.convert_to_tensor(filter_np)

#卷积层
    op = tf.nn.conv2d(fm_in, filter, strides=conv_stride_info, dilations=conv_dilation_info, padding=conv_padding_mode)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      # print("op:\n",sess.run(op))
      fm_out = sess.run(op)
      print("the tensorflow calc  fm_out's shape is",sess.run(tf.shape(op)))

    print("the theory     calc  fm_out's shape is " + str(fm_out_shape))


#文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

    np.savetxt(dir+'/conv_fm_in.txt' , fm_in_np.reshape(-1,1))
    np.savetxt(dir+'/conv_filter.txt', filter_np.reshape(-1,1))
    np.savetxt(dir+'/conv_fm_out.txt', fm_out.reshape(-1,1))

    print("==================================================")
    print("the fm_in filter_kernel fm_out is save in the file")
    print("==================================================")

    # 维度转化
    # filter_tranpose = filter_np.transpose((3,2,0,1))    #[ filter_height, filter_weight, in_channel, out_channels ]
    # fm_in_tranpose  = fm_in_np.transpose((0,3,1,2))     #[ batch, in_height, in_weight, in_channel ]===
    # fm_out_tranpose = fm_out.transpose((0,3,1,2))       #[ batch, in_height, in_weight, in_channel ]===
    # np.savetxt('fm_in_tranpose.txt' , fm_in_tranpose.reshape(-1,1) )
    # np.savetxt('filter_tranpose.txt', filter_tranpose.reshape(-1,1))
    # np.savetxt('fm_out_tranpose.txt', fm_out_tranpose.reshape(-1,1))


def cnn_operator_pool(fm_in_shape,pool_kernel_info,pool_stride_info,pool_mode):
#     pool_mode = "AVG"
#     fm_in_shape        = [1, 256, 256, 3]          # 输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
#     pool_kernel_info   = [1, 2,   2,  1]
#     pool_stride_info   = [1, 3,   3,  1]

    pool_padding_mode = 'VALID'
    # pool_padding_mode = 'SAME'
    if pool_padding_mode == 'VALID':
        pool_padding = [0, 0]
    elif pool_padding_mode == 'SAME':
        pool_padding = [
            (pool_kernel_info[1] - 1) / 2
           ,(pool_kernel_info[2] - 1) / 2
        ]
    else:
        print("padding mode error")
        sys.exit(0)
    print("conv_padding mode is ", str(pool_padding_mode))

#理论计算获取的输出特征图大小
    fm_out_shape =[
        1
        ,math.floor( (fm_in_shape[1]-pool_kernel_info[1]+2*pool_padding[0])/pool_stride_info[1]) + 1
        ,math.floor( (fm_in_shape[2]-pool_kernel_info[2]+2*pool_padding[1])/pool_stride_info[2]) + 1
        ,fm_in_shape[3]
    ]

#特征数据生成
    fm_in_np = np.random.randn(fm_in_shape[0], fm_in_shape[1], fm_in_shape[2], fm_in_shape[3]);
    fm_in = tf.convert_to_tensor(fm_in_np)

#池化层
    if pool_mode == "AVG":
        op = tf.nn.avg_pool(fm_in, pool_kernel_info, pool_stride_info, pool_padding_mode, "NHWC", name="avg_pool")
    elif (pool_mode == "MAX"):
        op = tf.nn.max_pool(fm_in, pool_kernel_info, pool_stride_info, pool_padding_mode, "NHWC", name="max_pool")
    else:
        print("pooling mode error")
        sys.exit(0)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      fm_out = sess.run(op)
      print("the tensorflow cal fm_out_pool's shape is",sess.run(tf.shape(op)))

    print("the theory     calc  fm_out_pool's shape is " + str(fm_out_shape))

# 文件保存
    dir = 'inout'
    if not os.path.exists(dir):
        os.mkdir(dir)

    np.savetxt(dir+'/pool_fm_in_'+pool_mode+'.txt', fm_in_np.reshape(-1, 1))
    np.savetxt(dir+'/pool_fm_out_'+pool_mode+'.txt', fm_out.reshape(-1, 1))

    print("==================================================")
    print("the fm_in filter_kernel fm_out is save in the file")
    print("==================================================")


