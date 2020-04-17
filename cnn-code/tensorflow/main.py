import cnn_operator as cnn_opt

if __name__ == '__main__':

#CONV operator
    fm_in_shape        = [1, 16, 16, 1]   #输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
    filter_shape       = [3, 3, 1, 1]      #卷积核参数 ====shape为 [ filter_height, filter_weight, in_channel, out_channels ]==
    conv_stride_info   = [1, 2, 2, 1]       #步进参数   ====shape为 [ 1, strides, strides, 1]，第一位和最后一位固定必须是1
    conv_dilation_info = [1, 1, 1, 1]
    conv_padding_mode  = "SAME"
    cnn_opt.cnn_operator_conv(fm_in_shape,filter_shape,conv_stride_info,conv_dilation_info,conv_padding_mode)

#POOL operator
    pool_mode = "MAX"
    fm_in_shape = [1, 256, 256, 3]  # 输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
    pool_kernel_info = [1, 2, 2, 1]
    pool_stride_info = [1, 3, 3, 1]

    # cnn_opt.cnn_operator_pool()