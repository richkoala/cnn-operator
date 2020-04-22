%% matlab�����������tensorflow��ѧϰ��ܣ���tensor��ά��˳����أ�
%% ������Ϊ����C���뿪������֤matlab��������ڴ�ͳ��ѧϰ��ܣ����Ի�ȡ����Ĺ�������

% dat = [1:1:100];
% BN_var = vpa(var(dat,0,2))    ˵��matlab�з�����㺯���м������0����ʾ���㷽��ʹ��1/(N-1)*(x-mean(x))^2
% BN_var = vpa(var(dat,1,2))    ˵��matlab�з�����㺯���м������1����ʾ���㷽��ʹ��1/N*(x-mean(x))^2
% me = mean(dat);
% er = dat-me;
% er_1 = (sum(er.^2));
% std_1= sqrt(1/(length(dat)-1) * er_1)
% var_1 =1/(length(dat)) * er_1;
% vpa(var_1)

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



fid_fm_in  = fopen('./inout/BN_fm_in.txt','r');
fid_fm_out = fopen('./inout/BN_fm_out.txt','r');

fm_in_shape        = [10, 3, 64, 64]     %#��������ͼ ====shapeΪ [ batch, in_height, in_weight, in_channel ]===
fm_out_shape       = fm_in_shape         %#�������ͼ ====shapeΪ [ batch, in_height, in_weight, in_channel ]===

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_in = permute(fm_in_tmp,[4,3,2,1]);
whos fm_in

% ����ʹ���������ͼ��С���չ�ʽ�����ȡ��������Դ��ѧϰ������ά����Ϣ
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2),fm_in_shape(1)]);
fm_out = permute(fm_out_tmp,[4,3,2,1]);
whos fm_out

%%��ֵ������㹫ʽ
fm_in_mean_var_dat = permute(fm_in,[2,1,3,4]);
fm_in_mean_var_dat = reshape(fm_in_mean_var_dat,3,[]);
BN_calc_data_num = fm_in_shape(4)*fm_in_shape(3)*fm_in_shape(1);
BN_mean = sum(fm_in_mean_var_dat,2)/BN_calc_data_num
BN_var = var(fm_in_mean_var_dat,0,2)

[Nb,Nof,Nor,Noc]   = size(fm_in);

gamma= ones(1,Nof);
beta = zeros(1,Nof);
eps = 0;


%% �������ͼ���㴦�� && ��չ�������
fm_mout = zeros(size(fm_out));

for nb=1:Nb
    for nof=1:Nof
        
        tmp = (sqrt(BN_var(nof))+eps)
        gamma_nif = gamma(nof);
        beta_nif  = beta(nof);
        
        for nor=1:Nor
            for noc=1:Noc
                nif = nof;
                nir = nor;
                nic = noc;
                fm_mout(nb,nof,nor,noc) = ( fm_in(nb,nif,nir,nic) - BN_mean(nif) ) / tmp * gamma_nif  + beta_nif ;
            end
        end
    end
end

err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)







