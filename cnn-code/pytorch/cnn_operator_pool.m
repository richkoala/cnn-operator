%% matlab�����������pytorch��ѧϰ��ܣ���tensor��ά��˳����أ�
%% ������Ϊ����C���뿪������֤matlab��������ڴ�ͳ��ѧϰ��ܣ����Ի�ȡ����Ĺ�������

%% ��������
%% 1��torch�ڴ���max_poolʱ���Խ���dilation���������max_poolû�ж�Ӧ��dilation�����ߵ��µ�����ߴ��СҲ���ڲ�ͬ����Ҫ���д���
%% 2��torch�ڴ���max_pool���ֶ�padding��������mask�����䲻����Ƚϣ�������ave_pooling�������������������


%%����֧��
% 1) stride 
% 2) kernel-size 
% 3) padding 
% 4��avg + max mode

clc
clear all
close all
fclose all

pool_mode = 1 % 0=max  1=average

if pool_mode == 0
    fid_fm_in  = fopen('./inout/pool_fm_in_MAX.txt','r');
    fid_fm_out = fopen('./inout/pool_fm_out_MAX.txt','r');
else
    fid_fm_in  = fopen('./inout/pool_fm_in_AVG.txt','r');
    fid_fm_out = fopen('./inout/pool_fm_out_AVG.txt','r');
end

fm_in_shape         = [1, 3, 256, 256, ]    % # ��������ͼ ====shapeΪ [ batch, in_channel, in_height, in_weight]===
pool_kernel_info    = [1, 1, 2, 2]
pool_stride_info    = [1, 1, 7, 7]
pool_dilation_info  = [1, 1, 3, 3]
pool_padding_info   = [1, 1]

Kr = pool_kernel_info(3);
Kc = pool_kernel_info(4);

Dr = pool_dilation_info(3);
Dc = pool_dilation_info(4);

%% �������ͼ�ߴ����
if pool_mode == 0  %max pooling
fm_out_shape =[ ...
    fm_in_shape(1) ...
    ,fm_in_shape(2) ...
    ,floor( (fm_in_shape(3)+2*pool_padding_info(1)-pool_dilation_info(3)*(pool_kernel_info(3)-1) - 1 )/pool_stride_info(3) + 1) ...
    ,floor( (fm_in_shape(4)+2*pool_padding_info(2)-pool_dilation_info(4)*(pool_kernel_info(4)-1) - 1 )/pool_stride_info(4) + 1) ...
]
else if pool_mode == 1 % average pooling
    fm_out_shape =[ ...
        fm_in_shape(1) ...
        ,fm_in_shape(2) ...
        ,floor( (fm_in_shape(3)+2*pool_padding_info(1)-pool_kernel_info(3) - 1)/pool_stride_info(3) + 1) ...
        ,floor( (fm_in_shape(4)+2*pool_padding_info(2)-pool_kernel_info(4)  -1)/pool_stride_info(4) + 1) ...
    ]        
    end
end

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2)]);
fm_in = permute(fm_in_tmp,[3,2,1]);
whos fm_in

% ����ʹ���������ͼ��С���չ�ʽ�����ȡ��������Դ��ѧϰ������ά����Ϣ
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2)]);
fm_out = permute(fm_out_tmp,[3,2,1]);
whos fm_out

[Nif,Nir,Nic]   = size(fm_in);
[Nof,Nor,Noc]   = size(fm_out);

fm_in_padding = zeros(Nif,Nir+pool_padding_info(1)*2,Nic+pool_padding_info(2)*2);
fm_in_padding(:,1+pool_padding_info(1):end-pool_padding_info(1),1+pool_padding_info(2):end-pool_padding_info(2))=fm_in;
[Nif,Nir,Nic]   = size(fm_in_padding);

%% �������ͼ���㴦�� && ��չ�������
fm_mout = zeros(size(fm_out));

for nof=1:Nof
    nif = nof;
    for nor=1:Nor
        for noc=1:Noc
            nir = (nor-1)*pool_stride_info(3)+1;
            nic = (noc-1)*pool_stride_info(4)+1;
            max_num = 0;
            max_dat = zeros(1,Kr*Kc);
            if (pool_mode==0)           %���pytorch��pooling
                
                for idx_r= nir:Dr:nir+(Kr-1)*Dr
                    for idx_c=nic:Dc:nic+(Kc-1)*Dc
%                         idx_r
%                         idx_c
                        if (idx_r==1 || idx_c==1 || idx_r== Nir || idx_c == Nic)
                        else
                            max_num = max_num +1;
                            max_dat(max_num)=fm_in_padding(nif,idx_r,idx_c);
                        end
                    end
                end
                fm_mout(nof,nor,noc) = max(max_dat(1:max_num));        
                
%               region = fm_in_padding(nif,nir:Dr:nir+(Kr-1)*Dr,nic:Dc:nic+(Kc-1)*Dc);   %%��Padding���㴦���е� 0 ����Ƚ�ʱ�Ĵ���       
%               fm_mout(nof,nor,noc)= max(max(region));                               
            else
                region = fm_in_padding(nif,nir:1:nir+(Kr-1),nic:1:nic+(Kc-1));
                fm_mout(nof,nor,noc)= sum(sum(region))/Kr/Kc;
            end
        end
    end
end

err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)




