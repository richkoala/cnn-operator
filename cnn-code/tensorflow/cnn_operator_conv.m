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
fid_filter = fopen('./inout/conv_filter.txt','r');
fid_fm_out = fopen('./inout/conv_fm_out.txt','r');

fm_in_shape        = [1, 16, 16, 1] %��������ͼ ====shapeΪ [ batch, in_height, in_weight, in_channel ]===
filter_shape       = [3, 3, 1, 1]   %����˲��� ====shapeΪ [ filter_height, filter_weight, in_channel, out_channels ]==
conv_stride_info   = [1, 2, 2, 1]   %��������   ====shapeΪ [ 1, strides, strides, 1]����һλ�����һλ�̶�������1
conv_dilation_info = [1, 1, 1, 1]
% conv_padding_mode = 0  %valid
conv_padding_mode = 1    %same 

Dr = conv_dilation_info(2);
Dc = conv_dilation_info(3);

if conv_padding_mode == 0
    conv_padding = [0, 0]
else if conv_padding_mode == 1
    conv_padding = [ ...
         (filter_shape(1) + (filter_shape(1)-1) * (conv_dilation_info(2) - 1) - 1) / 2 ...
        ,(filter_shape(2) + (filter_shape(2)-1) * (conv_dilation_info(3) - 1) - 1) / 2 ...
    ]
    else
    disp("padding mode error")
    end
end
disp(conv_padding)

%% �������ͼ�ߴ����
fm_out_shape =[ ...
    1 ...
    ,floor( (fm_in_shape(2)-(filter_shape(1)+(filter_shape(1)-1)*(conv_dilation_info(2)-1))+2*conv_padding(1))/conv_stride_info(2)) + 1 ...
    ,floor( (fm_in_shape(3)-(filter_shape(2)+(filter_shape(2)-1)*(conv_dilation_info(3)-1))+2*conv_padding(2))/conv_stride_info(3)) + 1 ...
    ,filter_shape(4) ...
]

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
filter_tmp = fscanf(fid_filter,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2)]);
fm_in = permute(fm_in_tmp,[1,3,2]);
whos fm_in

filter_tmp = reshape(filter_tmp,[filter_shape(4),filter_shape(3),filter_shape(2),filter_shape(1)]);
filter = permute(filter_tmp,[1,2,4,3]);
whos filter

% ����ʹ���������ͼ��С���չ�ʽ�����ȡ��������Դ��ѧϰ������ά����Ϣ
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2)]);
fm_out = permute(fm_out_tmp,[1,3,2]);
whos fm_out


[Nof,Nif,Kr,Kc] = size(filter);
[Nif,Nir,Nic]   = size(fm_in);
[Nof,Nor,Noc]   = size(fm_out);

fm_in_padding = zeros(Nif,Nir+conv_padding(1)*2,Nic+conv_padding(2)*2);
fm_in_padding(:,1+conv_padding:end-conv_padding,1+conv_padding:end-conv_padding)=fm_in;
[Nif,Nir,Nic]   = size(fm_in_padding);

%% �������ͼ���㴦�� && ��չ�������
fm_mout = zeros(size(fm_out));

for nof=1:Nof
    for nor=1:Nor
        for noc=1:Noc
            for nif=1:Nif
                for kr = 1:Kr
                    for kc=1:Kc
                        nir = (nor-1)*conv_stride_info(2)+1;
                        nic = (noc-1)*conv_stride_info(3)+1;
%                         nir = (nor)*conv_stride_info(2);
%                         nic = (noc)*conv_stride_info(3);
                        fm_mout(nof,nor,noc)=fm_mout(nof,nor,noc)+ fm_in_padding(nif,nir+(kr-1)*Dr,nic+(kc-1)*Dc)*filter(nof,nif,kr,kc);
                        disp([fm_in_padding(nif,nir+(kr-1)*Dr,nic+(kc-1)*Dc),filter(nof,nif,kr,kc),fm_mout(nof,nor,noc)])
                    end
                end
                disp("====")
            end
        end
    end
end

err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)




