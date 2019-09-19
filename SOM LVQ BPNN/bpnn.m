tic;
clear;
clc;

input_pca = [];
for i = 1 : 200
    str = int2str(i);
    str = strcat('baru/', str, '.jpg');
    img = imread(str);
    img_rgb = whos('img');
    size_rgb = img_rgb.bytes;
    img = im2double(img);
    if size_rgb == 4800
        img = rgb2gray(img);
    end
    [irow icol] = size(img);  
    temp = reshape(img, irow*icol, 1);
    %temp = transpose(temp);
    input_pca = [input_pca temp];
end
input_pca = transpose(input_pca);

pca_eigen_percent = 0.9;
pca_eigen_total = 0;

% ------PCA proses------

input_std = zscore(input_pca); 
input_cov = cov(input_std); 
[eigen_vectors eigen_values] = eig(input_cov); 

eigen_values = diag(eigen_values); 
eigen_values = abs(eigen_values); 
[eigen_values, index] = sortrows(eigen_values, -1); 

for i = 1 : 200
    pca_eigen_total = pca_eigen_total + eigen_values(i);
    if pca_eigen_total/sum(eigen_values) <= pca_eigen_percent
        pca_eigen_vectors(:, i) = eigen_vectors(:, uint16(index(i)));
    else
        break;
    end
end

pca_output = (pca_eigen_vectors.' * input_std.').'; 
[totalImage, pcaDim] = size(pca_output); 

clear i index pca_eigen_total pca_eigen_percent ;
clear pca_eigen_vectors eigen_values eigen_vectors;
clear irow icol img input_cov input_pca input_std;
clear str temp total_img data_choice img_rgb prompt;
clear size_rgb str_folder;
plot(pca_output(:,1),pca_output(:,pcaDim),'o');

inputlayer = pcaDim;
hidden = pcaDim*2;
output = 10;

alpha = 0.5;
miu = 0.4;
beta = 0.7*((hidden)^(1/inputlayer));
tr_percentage = 0.5;

train = totalImage * tr_percentage;
test = totalImage - train;

error_target = 0.01;
epoch_target = 100;

vij = zeros(inputlayer,hidden);
voj = zeros(1,hidden);
wjk = zeros(hidden,output);
wok = zeros(1,output);

init_method = input('Masukkan 0 untuk metode random. Masukkan angka lain untuk metode Nguyen-Widrow: ');

vij = rand(inputlayer,hidden) - 0.5;
wjk = rand(hidden,output) - 0.5;

if init_method == 0
    voj = rand(1,hidden) - 0.5;
    wok = rand(1,output) - 0.5;
else
    norm_vij = 0;
    for i = 1:inputlayer
        for j = 1:hidden
            norm_vij = norm_vij + (vij(i,j)^2);
        end
    end
    norm_vij = sqrt(norm_vij);

    norm_wjk = 0;
    for j = 1:hidden
        for k = 1: output
            norm_wjk = norm_wjk + (wjk(j,k)^2);
        end
    end
    norm_wjk = sqrt(norm_wjk);

    for i = 1:inputlayer
        for j = 1:hidden
            vij(i,j) = (beta * vij(i,j))/norm_vij;
        end
    end

    for j = 1:hidden
        for k = 1:output
            wjk(j,k) = (beta * wjk(j,k))/norm_wjk;
        end
    end

    voj = (beta+beta)*rand(1, hidden) - beta; 
    wok = (beta+beta)*rand(1, output) - beta;
end

deltaij = zeros(inputlayer,hidden);
deltajk = zeros(hidden,output);
deltaoj = zeros(hidden);
deltaok = zeros(output);

for n = 1:totalImage
    for i = 1:pcaDim
        scaled_inputlayer(n,i) = (pca_output(n,i)-min(pca_output(:,i))) / (max(pca_output(:,i))-min(pca_output(:,i)));
    end
end

in = zeros(totalImage,inputlayer);
target = zeros(totalImage,output);
for i = 1:output
    in(i:output:totalImage,:) = scaled_inputlayer(((i-1)*totalImage/output)+1:i*totalImage/output,:);
    target(i:output:totalImage,i) = 1;
end
clear scaled_inputlayer;

for n = 1:train
    data_train(n,:) = in(n,:);
    train_target(n,:) = target(n,:);
end

epoch = 0;

while true
    epoch = epoch + 1;
    error_train = zeros(train,1);
    
    for iter = 1:train
        for i = 1:inputlayer
            x(i) = data_train(iter,i);
        end
        
        for j = 1:hidden
            z_in(j) = voj(j);
            for i = 1:inputlayer
                z_in(j) = z_in(j) + (x(i) * vij(i,j));
            end
            z(j) = 1/(1 + exp(-z_in(j)));
        end
        
        for k = 1:output
            y_in(k) = wok(k);
            for j = 1:hidden
                y_in(k) = y_in(k) + (z(j) * wjk(j,k));
            end
            y(k) = 1/(1 + exp(-y_in(k)));
        end
        
        for n = 1:output
            error_train(iter) = error_train(iter) + (0.5 * (train_target(iter,n) - y(n))^2);
        end
        
        for k = 1:output
            dk(k) = -(train_target(iter,k)-y(k)) * y(k) * (1 - y(k));
        end
        
        for j = 1:hidden
            for k = 1:output
                deltajk(j,k) = alpha * dk(k) * z(j) + miu * deltajk(j,k);
            end
        end
        
        for k = 1:output
            deltaok(k) = alpha * dk(k) + miu * deltaok(k);
        end
        
        for j = 1:hidden
            din_j(j) = 0;
            for k = 1:output
                din_j(j) = din_j(j) + dk(k) * wjk(j,k);
            end
            dj(j) = din_j(j) * z(j) * (1 - z(j));
        end
        
        for i = 1:inputlayer
            for j = 1:hidden
                deltaij(i,j) = alpha * dj(j) * x(i) + miu * deltaij(i,j);
            end
        end
        
        for j = 1:hidden
            deltaoj(j) = alpha * dj(j) + miu * deltaoj(j);
        end
        
        for j = 1:hidden
            for k = 1:output
                wjk(j,k) = wjk(j,k) - deltajk(j,k);
            end
        end
        
        for k = 1:output
            wok(k) = wok(k) - deltaok(k);
        end
        
        for i = 1:inputlayer
            for j = 1:hidden
                vij(i,j) = vij(i,j) - deltaij(i,j);
            end
        end
        
        for j = 1:hidden
            voj(j) = voj(j) - deltaoj(j);
        end
    end
    
    err_avg(epoch) = mean(error_train);
    
    if err_avg(epoch) <= error_target || epoch == epoch_target
        break;
    end
end

for n = 1:test
    data_test(n,:) = in(n+test,:);
    test_target(n,:) = target(n+test,:);
end

recognition = zeros(test,1);

for iter = 1:test
    for i = 1:inputlayer
        x(i) = data_test(iter,i);
    end

    benar = 0;
    salah = 0;
    
    for j = 1:hidden
        z_in(j) = voj(j);
        for i = 1:inputlayer
            z_in(j) = z_in(j) + (x(i) * vij(i,j));
        end
        z(j) = 1/(1 + exp(-z_in(j)));
    end

    for k = 1:output
        y_in(k) = wok(k);
        for j = 1:hidden
            y_in(k) = y_in(k) + (z(j) * wjk(j,k));
        end
        y(k) = 1/(1 + exp(-y_in(k)));
    end
    
    for i = 1:output
        if y(i) == max(y)
            y_tmp(1, i) = 1;
        else
            y_tmp(1, i) = 0;
        end
    end
    
    for i = 1:output
        if y_tmp(1,i) == test_target(iter,i)
            benar = benar + 1;
        else
            salah = salah + 1;
        end
    end
    
    if benar == 10
        recognition(iter) = 1;
    end
end

l = 1:epoch;
plot(l,err_avg);
xlabel('Epoch');
ylabel('Error');

rr = sum(recognition)/test;
fprintf('Epoch training = %d\n', epoch); 
fprintf('Error min = %f\n', min(err_avg));
fprintf('Error max = %f\n', max(err_avg));
fprintf('Recognition rate = %.2f\n', rr*100);
toc;