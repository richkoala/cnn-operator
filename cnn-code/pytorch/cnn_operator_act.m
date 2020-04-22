%% matlab代码仅适用于tensorflow的学习框架，与tensor的维度顺序相关，
%% 可以作为后续C代码开发的验证matlab程序，相对于传统的学习框架，可以获取更多的过程数据


%% Prelu 为带训练参数的RELU，这里仅考虑推理状态，设置为固定值进行判断，对于小于的输入使用

clc
clear all
close all
fclose all

fid_fm_in            = fopen('./inout/act_fm_in.txt','r');
fid_fm_out_relu      = fopen('./inout/act_fm_out_relu.txt','r');    
fid_fm_out_prelu     = fopen('./inout/act_fm_out_prelu.txt','r');    
fid_fm_out_leakyrelu = fopen('./inout/act_fm_out_leakyrelu.txt','r'); 
fid_fm_out_relu6     = fopen('./inout/act_fm_out_relu6.txt','r'); 
fid_fm_out_sigmoid   = fopen('./inout/act_fm_out_sigmoid.txt','r'); 
fid_fm_out_Tanh      = fopen('./inout/act_fm_out_Tanh.txt','r'); 

fm_in_shape        = [1, 3, 256, 256,];     %#输入特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
fm_out_shape       = fm_in_shape;         %#输出特征图 ====shape为 [ batch, in_height, in_weight, in_channel ]===
prelu_param        = 0.28;
leakyrelu_param    = 1e-2;

fm_in_tmp           = fscanf(fid_fm_in ,'%e',inf);
fm_out_relu_tmp     = fscanf(fid_fm_out_relu,'%e',inf);
fm_out_prelu_tmp    = fscanf(fid_fm_out_prelu,'%e',inf);
fm_out_leakyrelu_tmp= fscanf(fid_fm_out_leakyrelu,'%e',inf);
fm_out_relu6_tmp    = fscanf(fid_fm_out_relu6,'%e',inf);
fm_out_sigmoid_tmp  = fscanf(fid_fm_out_sigmoid,'%e',inf);
fm_out_Tanh_tmp     = fscanf(fid_fm_out_Tanh,'%e',inf);

fm_in_tmp           = reshape(fm_in_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_in = permute(fm_in_tmp,[4,3,2,1]);
whos fm_in

% 这里使用输出特征图大小按照公式计算获取，并非来源于学习框架输出维度信息
fm_out_relu_tmp     = reshape(fm_out_relu_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_out_prelu_tmp    = reshape(fm_out_prelu_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_out_leakyrelu_tmp= reshape(fm_out_leakyrelu_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_out_relu6_tmp    = reshape(fm_out_relu6_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_out_sigmoid_tmp  = reshape(fm_out_sigmoid_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);
fm_out_Tanh_tmp     = reshape(fm_out_Tanh_tmp,[fm_in_shape(4),fm_in_shape(3),fm_in_shape(2),fm_in_shape(1)]);

fm_out_relu = permute(fm_out_relu_tmp,[4,3,2,1]);
fm_out_prelu = permute(fm_out_prelu_tmp,[4,3,2,1]);
fm_out_leakyrelu = permute(fm_out_leakyrelu_tmp,[4,3,2,1]);
fm_out_relu6 = permute(fm_out_relu6_tmp,[4,3,2,1]);
fm_out_sigmoid = permute(fm_out_sigmoid_tmp,[4,3,2,1]);
fm_out_Tanh = permute(fm_out_Tanh_tmp,[4,3,2,1]);

%%均值方差计算公式

[Nb,Nof,Nor,Noc]   = size(fm_in);
%% 输出特征图清零处理 && 开展卷积计算
fm_mout_relu = zeros(size(fm_out_relu));
fm_mout_prelu = zeros(size(fm_out_relu));
fm_mout_leakyrelu = zeros(size(fm_out_relu));
fm_mout_relu6 = zeros(size(fm_out_relu));
fm_mout_sigmoid = zeros(size(fm_out_relu));
fm_mout_Tanh = zeros(size(fm_out_relu));


for nb=1:Nb
    for nof=1:Nof
        nif = nof;
        for nor=1:Nor
            nir = nor;
            for noc=1:Noc
                nic = noc;
                if ( fm_in(nb,nif,nir,nic) >= 0 )
                    fm_mout_relu(nb,nof,nor,noc)    = fm_in(nb,nif,nir,nic) ;
                    fm_mout_prelu(nb,nof,nor,noc)    = fm_in(nb,nif,nir,nic) ; 
                    fm_mout_leakyrelu(nb,nof,nor,noc)= fm_in(nb,nif,nir,nic) ;
                    if (fm_in(nb,nif,nir,nic) > 6) 
                        fm_mout_relu6(nb,nof,nor,noc)=6 ;
                    else
                        fm_mout_relu6(nb,nof,nor,noc)=fm_in(nb,nif,nir,nic);
                    end
                else
                    fm_mout_relu(nb,nof,nor,noc) = 0;
                    fm_mout_prelu(nb,nof,nor,noc) = prelu_param*fm_in(nb,nif,nir,nic);
                    fm_mout_leakyrelu(nb,nof,nor,noc)= fm_in(nb,nif,nir,nic)*leakyrelu_param;
                    fm_mout_relu6(nb,nof,nor,noc) = 0;
                end
                
                fm_mout_sigmoid(nb,nof,nor,noc) = 1/(1+exp(-fm_in(nb,nif,nir,nic)));
                fm_mout_Tanh(nb,nof,nor,noc) = (exp(fm_in(nb,nif,nir,nic)) - exp(-fm_in(nb,nif,nir,nic))) /  (exp(fm_in(nb,nif,nir,nic)) + exp(-fm_in(nb,nif,nir,nic)));
            end
        end
    end
end

err_relu      = (fm_mout_relu - fm_out_relu);
err_prelu     = (fm_mout_prelu - fm_out_prelu);
err_leakyrelu = (fm_mout_leakyrelu - fm_out_leakyrelu);
err_relu6     = (fm_mout_relu6 - fm_out_relu6);
err_sigmoid   = (fm_mout_sigmoid - fm_out_sigmoid);
err_Tanh      = (fm_mout_Tanh - fm_out_Tanh);

err_relu= reshape(err_relu,[],1);
err_prelu= reshape(err_prelu,[],1);
err_leakyrelu= reshape(err_leakyrelu,[],1);
err_relu6= reshape(err_relu6,[],1);
err_sigmoid= reshape(err_sigmoid,[],1);
err_Tanh= reshape(err_Tanh,[],1);

subplot(621);plot(err_relu);title('err\_relu')
subplot(622);plot(err_prelu);title('err\_prelu')
subplot(623);plot(err_leakyrelu);title('err\_leakyrelu')
subplot(624);plot(err_relu6);title('err\_relu6')
subplot(625);plot(err_sigmoid);title('err\_sigmoid')
subplot(626);plot(err_Tanh);title('err\_Tanh')







