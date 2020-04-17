%% matlab代码仅适用于tensorflow的学习框架，与tensor的维度顺序相关，
%% 可以作为后续C代码开发的验证matlab程序，相对于传统的学习框架，可以获取更多的过程数据

%%参数支持
% 1) stride 
% 2) kernel-size 
% 3) padding 
% 4）avg + max mode

clc
clear all
close all
fclose all

pool_mode = 0 %max
% pool_mode = 1 %average

if pool_mode == 0
    fid_fm_in  = fopen('./pool_fm_in_MAX.txt','r');
    fid_fm_out = fopen('./pool_fm_out_MAX.txt','r');
else
    fid_fm_in  = fopen('./pool_fm_in_AVG.txt','r');
    fid_fm_out = fopen('./pool_fm_out_AVG.txt','r');
end

fm_in_shape = [1, 256, 256, 3]  % 输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
pool_kernel_info = [1,2,2,1]
pool_stride_info = [1,3,3,1]
pool_dilation_info = [1, 1, 1, 1]

Kr = pool_kernel_info(2);
Kc = pool_kernel_info(3);

Dr = pool_dilation_info(2);
Dc = pool_dilation_info(3);

pool_padding_mode = 0     %valid mode
% pool_padding_mode = 1       % same mode
if pool_padding_mode == 0
    pool_padding = [0, 0]
else if pool_padding_mode == 1
    pool_padding = [ ...
         (pool_kernel_info(2) - 1) / 2 ...
        ,(pool_kernel_info(3) - 1) / 2 ...
    ]
    else
    disp("padding mode error")
    end
end
disp(pool_padding)

%% 输出特征图尺寸计算
fm_out_shape =[ ...
    1 ...
    ,floor( (fm_in_shape(2)-pool_kernel_info(2)+2*pool_padding(1))/pool_stride_info(2)) + 1 ...
    ,floor( (fm_in_shape(3)-pool_kernel_info(3)+2*pool_padding(2))/pool_stride_info(3)) + 1 ...
    ,fm_in_shape(4) ...
]

fm_in_tmp  = fscanf(fid_fm_in ,'%e',inf);
fm_out_tmp = fscanf(fid_fm_out,'%e',inf);

fm_in_tmp = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2)]);
fm_in = permute(fm_in_tmp,[1,3,2]);
whos fm_in

% 这里使用输出特征图大小按照公式计算获取，并非来源于学习框架输出维度信息
fm_out_tmp = reshape(fm_out_tmp,[fm_out_shape(4),fm_out_shape(3),fm_out_shape(2)]);
fm_out = permute(fm_out_tmp,[1,3,2]);
whos fm_out

[Nif,Nir,Nic]   = size(fm_in);
[Nof,Nor,Noc]   = size(fm_out);

fm_in_padding = zeros(Nif,Nir+pool_padding(1)*2,Nic+pool_padding(2)*2);
fm_in_padding(:,1+pool_padding:end-pool_padding,1+pool_padding:end-pool_padding)=fm_in;
[Nif,Nir,Nic]   = size(fm_in_padding);

%% 输出特征图清零处理 && 开展卷积计算
fm_mout = zeros(size(fm_out));

for nof=1:Nof
    nif = nof;
    for nor=1:Nor
        for noc=1:Noc
            nir = (nor-1)*pool_stride_info(2)+1;
            nic = (noc-1)*pool_stride_info(3)+1;
            region = fm_in_padding(nif,nir:Dr:nir+(Kr-1),nic:Dc:nic+(Kc-1));
            if (pool_mode==0)
                fm_mout(nof,nor,noc)= max(max(region));
            else
                fm_mout(nof,nor,noc)= sum(sum(region))/Kr/Kc;
            end
        end
    end
end

err = (fm_mout - fm_out);
err= reshape(err,[],1);
plot(err)




