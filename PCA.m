Data =[110 8 5; 228 21 3; 342 31 14; 375 40 15; 578 48 4; 699 60 12; 807 71 14; 929 79 16; 1040 92 18; 1160 101 38; 1262 109 28; 1376 121 32; 1499 128 35; 1620 143 28; 1722 150 30; 1833 159 15; 1948 172 12; 2077 181 35; 2282 190 23; 2999 202 30];
plot3(Data(:,1),Data(:,2),Data(:,3),'o')
labels = 'abcdefghijklmnopqrst';
labels = 'labels';
text(Data(:,1), Data(:,2), Data(:,3), labels, 'horizontal','left', 'vertical','bottom');
xlabel('X');
ylabel('Y');
zlabel('Z');

%standarisasi data
DatStd=zscore(Data);
hold on;
figure('name','Standardized Data');
plot3(DatStd(:,1),DatStd(:,2),DatStd(:,3),'o')
text(DatStd(:,1), DatStd(:,2), DatStd(:,3), labels, 'horizontal','left', 'vertical','bottom');
xlabel('X');
ylabel('Y');
zlabel('Z');

%menghitung covarians dari data yang sudah distandarisasi
C=cov(DatStd);

hold on;
figure('name','Covariance between X and X');
plot(DatStd(:,1),DatStd(:,1),'o');
text(DatStd(:,1), DatStd(:,1), labels, 'horizontal','left', 'vertical','bottom');
xlabel('X');
ylabel('X');

hold on;
figure('name','Covariance between X and Y');
plot(DatStd(:,1),DatStd(:,2),'o');
text(DatStd(:,1), DatStd(:,2), labels, 'horizontal','left', 'vertical','bottom');
xlabel('X');
ylabel('Y');

hold on;
figure('name','Covariance between X and Z');
plot(DatStd(:,1),DatStd(:,3),'o');
text(DatStd(:,1), DatStd(:,3), labels, 'horizontal','left', 'vertical','bottom');
xlabel('X');
ylabel('Z');

hold on;
figure('name','Covariance between Y and Z');
plot(DatStd(:,2),DatStd(:,3),'o');
text(DatStd(:,2), DatStd(:,3), labels, 'horizontal','left', 'vertical','bottom');
xlabel('Y');
ylabel('Z');

%eigen value decomposition
[V,D] = eig(C);

%ubah letak eigenvektor berdasarkan eigenvalue (descending order)
D2=diag(sort(diag(D),'descend'));
[c, ind]=sort(diag(D),'descend');
V2=V(:,ind);
figure('name','principal component plot')
plot3(DatStd(:,1), DatStd(:,2),DatStd(:,3), 'o');
text(DatStd(:,1), DatStd(:,2),DatStd(:,3), labels, 'horizontal','left', 'vertical','bottom');
hold all
plot3(V2(1,1)*[-5 5], V2(2,1)*[-5 5], V2(3,1)*[-5 5], '-r')
hold all
plot3(V2(1,2)*[-2 2], V2(2,2)*[-2 2], V2(3,2)*[-2 2], '-r')
xlabel('X');
ylabel('Y');
zlabel('Z');

figure('name','axis changed based on principal components')
PcaPos = DatStd * V2(:, 1:2);
plot(PcaPos(:,1),PcaPos(:,2),'o');
text(PcaPos(:,1),PcaPos(:,2), labels, 'horizontal','left', 'vertical','bottom');
xlabel('PC1');
ylabel('PC2');

%source: https://tyangluhtu.wordpress.com/2013/04/19/langkah-umum-principal-component-analysis/amp/