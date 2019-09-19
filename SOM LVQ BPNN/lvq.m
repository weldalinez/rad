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
    input_pca = [input_pca temp];
end
input_pca = transpose(input_pca);

pca_eigen_percent = 0.9;
pca_eigen_total = 0;

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

fitur = pcaDim;
cluster = 10;

train = 0.5 * totalImage;
test = 0.5 * totalImage;

alpha = 0.5;
c = 0.5;
alpha_target = 0.01;

input = zeros(totalImage, fitur);
for i = 1:cluster
    input(i:cluster:totalImage,:) = pca_output(((i-1)*totalImage/cluster)+1:i*totalImage/cluster,:);
    target(i:cluster:totalImage) = i;
end

for n = 1:train
    trainInput(n, :) = input(n, :);
    trainTarget(n) = target(n);
end

for j = 1:cluster
    Vp(j, :) = trainInput((randi(train/cluster)-1)*cluster+j, :);
end

epoch = 0;

while(true)
    epoch = epoch + 1;

    for iter = 1:train
        d = zeros(cluster, 1);
        for j = 1:cluster
            for i = 1:fitur
                d(j) = d(j) + (Vp(j,i) - trainInput(iter,i))^2;
            end
            d(j) = d(j)^0.5;
        end
        
        [d_min, d_min_index] = min(d);
        
        if trainTarget(iter) == d_min_index
            Vp(d_min_index) = Vp(d_min_index) + (alpha*(trainInput(iter)-Vp(d_min_index)));
        else
            Vp(d_min_index) = Vp(d_min_index) - (alpha*(trainInput(iter)-Vp(d_min_index)));
            Vp(trainTarget(iter)) = Vp(trainTarget(iter)) + (alpha*(trainInput(iter)-Vp(trainTarget(iter))));
        end
    end
    
    alpha = alpha * c;
    
    if alpha <= alpha_target
        break;
    end
end

for n = 1:test
    testInput(n, :) = input(n+train, :);
    testTarget(n) = target(n+train);
end
recognition = zeros(1, test);

for iter = 1:test
    d = zeros(cluster,1);
    for j = 1:cluster
        for i = 1:fitur
            d(j) = d(j) + (Vp(j,i) - testInput(iter,i))^2;
        end
        d(j) = d(j)^0.5;
    end
        
    [d_min, d_min_index] = min(d);
    
    if testTarget(iter) == d_min_index
        recognition(iter) = 1;
    end
end
recognition_rate = mean(recognition);
toc;