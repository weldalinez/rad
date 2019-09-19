%Weldaline - 1506673800

input = csvread('datairis.csv');

[rows,columns] = size(input);  
input(1:10,columns+1)=1;
input(11:20,columns+1)=2;
input(21:30,columns+1)=3;
input(31:40,columns+1)=4;
input(41:50,columns+1)=5;
input(51:60,columns+1)=6;
input(61:70,columns+1)=7;
input(71:80,columns+1)=8;
input(81:90,columns+1)=9;
input(91:100,columns+1)=10;
input(101:110,columns+1)=11;
input(111:120,columns+1)=12;
input(121:130,columns+1)=13;
input(131:140,columns+1)=14;
input(141:150,columns+1)=15;

data(1:15:rows,:) = input(1:10,:);
data(2:15:rows,:) = input(11:20,:);
data(3:15:rows,:) = input(21:30,:);
data(4:15:rows,:) = input(31:40,:);
data(5:15:rows,:) = input(41:50,:);
data(6:15:rows,:) = input(51:60,:);
data(7:15:rows,:) = input(61:70,:);
data(8:15:rows,:) = input(71:80,:);
data(9:15:rows,:) = input(81:90,:);
data(10:15:rows,:) = input(91:100,:);
data(11:15:rows,:) = input(101:110,:);
data(12:15:rows,:) = input(111:120,:);
data(13:15:rows,:) = input(121:130,:);
data(14:15:rows,:) = input(131:140,:);
data(15:15:rows,:) = input(141:150,:);

x = columns;                          % input
y = 15;                               % cluster

vp = data(1:y,1:columns);
 
trainttl = floor(0.5*rows);
testttl = floor(0.5*rows);

traininp = data(1:trainttl,1:columns);
testinp = data(trainttl+1:rows,1:columns); % untuk testing
tkTest = data(trainttl+1:rows,columns+1); % untuk testing

a = 0.4;                   % laju pembelajaran
c = 0.2;                     % konstanta penurunan laju pembelajaran
d = zeros(y,1);                % jarak euclidean
epoch = 0;    
benar = 0;

% Training
while true
    epoch = epoch + 1;
    
    for count = 1:trainttl
        for y = 1:y
            for x = 1:x
                d(y) = d(y) + ((traininp(count,x) - vp(y,x))^2);
            end
            d(y) = sqrt(d(y));
        end
        
        minimum = min(d); % vektor pewakil
        
        for y = 1:y
            if d(y)== minimum
                indeks = y;
            end
        end
        
        for x = 1:x
            vp(indeks,x) = vp(indeks,x) + (a * (traininp(count,x) - vp(indeks,x))); % hitung Vp baru
        end
    end
    
    a = c * a; %a baru
    
    if a <= 0.01
        break;
    end
end

% Testing
for count = 1:testttl
    
    for y = 1:y
        for x = 1:x
            d(y) = d(y) + ((testinp(count,x) - vp(y,x))^2);
        end
        d(y) = sqrt(d(y));
    end
    
    minimum = min(d); % vektor pewakil
    
    for y = 1:y
        if d(y)== minimum
            indeks(count,1) = y;
        end
    end
    
     if indeks(count)== tkTest(count)
        benar = benar+1;
    end
end

recog_rate = benar/testttl; 
disp(recog_rate* 100);
