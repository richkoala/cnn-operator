import cnn_operator as cnn_opt

if __name__ == '__main__':

#CONV operator
    fm_in_shape        = [1, 11 ,176, 176]     #输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
    kernel_shape       = [10, 11, 5, 5]      #卷积核参数 ====shape为 [ out_channels,in_channel, filter_height, filter_weight]==
    conv_stride_info   = [1, 1, 7, 7]      #步进参数   ====shape为 [ 1, 1, row_strides, col_strides]
    conv_dilation_info = [1, 1, 3, 3]
    conv_padding_info  = [13,13]
    # cnn_opt.cnn_operator_conv(fm_in_shape,kernel_shape,conv_stride_info,conv_dilation_info,conv_padding_info)



#POOL operator
    pool_mode = "AVG"
    fm_in_shape         = [1, 3, 256, 256, ]  # 输入特征图 ====shape为 [ batch, in_channel, in_height, in_weight]===
    pool_kernel_info    = [1, 1, 2, 2]
    pool_stride_info    = [1, 1, 7, 7]
    pool_dilation_info  = [1, 1, 3, 3]
    pool_padding_info   = [1, 1]

    cnn_opt.cnn_operator_pool(fm_in_shape,pool_kernel_info,pool_stride_info,pool_dilation_info,pool_padding_info,pool_mode)

