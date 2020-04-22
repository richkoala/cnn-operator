%% matlab�����������pytorch��ѧϰ��ܣ���tensor��ά��˳����أ�
%% ������Ϊ����C���뿪������֤matlab��������ڴ�ͳ��ѧϰ��ܣ����Ի�ȡ����Ĺ�������

%% FC ȫ���Ӵ������
% batch_size
% input_node 
% output_node

clc
clear all
close all
fclose all

batch_size = 10;
fm_in_num = 20;
fm_out_num = 30;

fid_fm_in  = fopen('./inout/fc_fm_in.txt','r');
fid_weight = fopen('./inout/fc_weight.txt','r');
fid_bias   = fopen('./inout/fc_bias.txt','r');
fid_fm_out = fopen('./inout/fc_fm_out.txt','r');

fm_in_shape         = [batch_size,  fm_in_num]    % # ��ڵ�����
weight_shape        = [fm_out_num,  fm_in_num];   % ȫ����Ȩ�ز���
fm_out_shape        = [batch_size,  fm_out_num];   % ȫ����Ȩ�ز���

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
weight_tmp  = fscanf(fid_weight ,'%e',inf);
bias_tmp = fscanf(fid_bias,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in = reshape(fm_in_tmp,[fm_in_shape(2),fm_in_shape(1)]);
whos fm_in

weight_tmp = reshape(weight_tmp,[weight_shape(2),weight_shape(1)]);
weight = permute(weight_tmp,[2,1]);
whos weight

bias = repmat(bias_tmp, 1, batch_size);
whos bias

fm_out = reshape(fm_out_tmp,[fm_out_shape(2),fm_out_shape(1)]);
whos fm_out

%% ��չȫ���Ӽ���
fm_mout = weight * fm_in + bias;


err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)




