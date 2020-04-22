%% matlab代码仅适用于tensorflow的学习框架，与tensor的维度顺序相关，
%% 可以作为后续C代码开发的验证matlab程序，相对于传统的学习框架，可以获取更多的过程数据

% dat = [1:1:100];
% BN_var = vpa(var(dat,0,2))    说明matlab中方差计算函数中间参数“0”表示计算方差使用1/(N-1)*(x-mean(x))^2
% BN_var = vpa(var(dat,1,2))    说明matlab中方差计算函数中间参数“1”表示计算方差使用1/N*(x-mean(x))^2
% me = mean(dat);
% er = dat-me;
% er_1 = (sum(er.^2));
% std_1= sqrt(1/(length(dat)-1) * er_1)
% var_1 =1/(length(dat)) * er_1;
% vpa(var_1)

%%参数支持
% 1) stride 
% 2) kernel-size 
% 3) padding 
% 4）dilation

%%参数暂支持
% 5）depth-wise

clc
clear all
close all
fclose all



fid_fm_in  = fopen('./inout/BN_fm_in.txt','r');
fid_fm_out = fopen('./inout/BN_fm_out.txt','r');

fm_in_shape        = [10, 3, 64, 64]     %#输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
fm_out_shape       = fm_in_shape         %#输出特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_in = permute(fm_in_tmp,[4,3,2,1]);
whos fm_in

% 这里使用输出特征图大小按照公式计算获取，并非来源于学习框架输出维度信息
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2),fm_in_shape(1)]);
fm_out = permute(fm_out_tmp,[4,3,2,1]);
whos fm_out

%%均值方差计算公式
fm_in_mean_var_dat = permute(fm_in,[2,1,3,4]);
fm_in_mean_var_dat = reshape(fm_in_mean_var_dat,3,[]);
BN_calc_data_num = fm_in_shape(4)*fm_in_shape(3)*fm_in_shape(1);
BN_mean = sum(fm_in_mean_var_dat,2)/BN_calc_data_num
BN_var = var(fm_in_mean_var_dat,0,2)

[Nb,Nof,Nor,Noc]   = size(fm_in);

gamma= ones(1,Nof);
beta = zeros(1,Nof);
eps = 0;


%% 输出特征图清零处理 && 开展卷积计算
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







