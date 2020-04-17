%% matlab�����������tensorflow��ѧϰ��ܣ���tensor��ά��˳����أ�
%% ������Ϊ����C���뿪������֤matlab��������ڴ�ͳ��ѧϰ��ܣ����Ի�ȡ����Ĺ�������

%%����֧��
% 1) stride 
% 2) kernel-size 
% 3) padding 
% 4��dilation

%%������֧��
% 5��depth-wise

clc
clear all
close all
fclose all

fid_fm_in  = fopen('./inout/conv_fm_in.txt','r');
fid_weight = fopen('./inout/conv_weight.txt','r');
fid_bias   = fopen('./inout/conv_bias.txt','r');
fid_fm_out = fopen('./inout/conv_fm_out.txt','r');

fm_in_shape        = [1, 11 ,176, 176]     %#��������ͼ ====shapeΪ [ batch, in_height, in_weight, in_channel ]===
kernel_shape       = [10, 11, 5, 5 ]      %#����˲��� ====shapeΪ [ out_channels,in_channel, filter_height, filter_weight]==
conv_stride_info   = [1, 1, 7, 7 ]      %#��������   ====shapeΪ [ 1, 1, row_strides, col_strides]
conv_dilation_info = [1, 1, 3, 3]
conv_padding_info  = [13,13]

Dr = conv_dilation_info(3);
Dc = conv_dilation_info(4);

%% �������ͼ�ߴ����
fm_out_shape = [
    1
    ,kernel_shape(1)
    ,floor((fm_in_shape(3)+2*conv_padding_info(1)-conv_dilation_info(3)*(kernel_shape(3)-1)-1)/conv_stride_info(3)+1)
    ,floor((fm_in_shape(4)+2*conv_padding_info(2)-conv_dilation_info(4)*(kernel_shape(4)-1)-1)/conv_stride_info(4)+1)
]

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
weight_tmp = fscanf(fid_weight,'%e',inf);
bias_tmp   = fscanf(fid_bias  ,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2)]);
fm_in = permute(fm_in_tmp,[3,2,1]);
whos fm_in

weight_tmp = reshape(weight_tmp,[kernel_shape(4),kernel_shape(3),kernel_shape(2),kernel_shape(1)]);
weight = permute(weight_tmp,[4,3,2,1]);
whos weight

bias = bias_tmp
whos bias

% ����ʹ���������ͼ��С���չ�ʽ�����ȡ��������Դ��ѧϰ������ά����Ϣ
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2)]);
fm_out = permute(fm_out_tmp,[3,2,1]);
whos fm_out


[Nof,Nif,Kr,Kc] = size(weight);
[Nif,Nir,Nic]   = size(fm_in);
[Nof,Nor,Noc]   = size(fm_out);

fm_in_padding = zeros(Nif,Nir+conv_padding_info(1)*2,Nic+conv_padding_info(2)*2);
fm_in_padding(:,1+conv_padding_info(1):end-conv_padding_info(1),1+conv_padding_info(2):end-conv_padding_info(2))=fm_in;
[Nif,Nir,Nic]   = size(fm_in_padding);

%% �������ͼ���㴦�� && ��չ�������
fm_mout = zeros(size(fm_out));

for nof=1:Nof
    for nor=1:Nor
        for noc=1:Noc
            for nif=1:Nif
                for kr = 1:Kr
                    for kc=1:Kc
                        nir = (nor-1)*conv_stride_info(3)+1;
                        nic = (noc-1)*conv_stride_info(4)+1;
                        fm_mout(nof,nor,noc)=fm_mout(nof,nor,noc)+ fm_in_padding(nif,nir+(kr-1)*Dr,nic+(kc-1)*Dc)*weight(nof,nif,kr,kc);
%                         disp([fm_in_padding(nif,nir+(kr-1)*Dr,nic+(kc-1)*Dc),weight(nof,nif,kr,kc),fm_mout(nof,nor,noc)]);
                    end
                end
            end
            fm_mout(nof,nor,noc) = fm_mout(nof,nor,noc) + bias(nof);
%             disp('====')
        end
    end
end

err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)




