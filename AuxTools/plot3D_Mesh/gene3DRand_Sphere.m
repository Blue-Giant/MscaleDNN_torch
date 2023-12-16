clc;
clear all
close all

num2radius = 50;
num2theta = 40;
num2gamma = 30;

XX = zeros(num2radius,num2theta, num2gamma);
YY = zeros(num2radius,num2theta, num2gamma);
ZZ = zeros(num2radius,num2theta, num2gamma);

r1 = 1.0;
r2 = 2.0;
rho = r1 + (r2-r1)*rand(num2radius, 1);
theta = 0+2*pi*rand(num2theta, 1);
gamma = 0+pi*rand(num2gamma, 1);

for i=1:num2radius
    for j=1:num2theta
        XX(i,j,1:end) = rho(i)*sin(theta(j)).*cos(gamma);
        YY(i,j,1:end) = rho(i)*sin(theta(j)).*sin(gamma);
        ZZ(i,j,1:end) = rho(i)*cos(theta(j));
    end
end

XX2inner = XX(1:num2radius,:,:);
YY2inner = YY(1:num2radius,:,:);
ZZ2inner = ZZ(1:num2radius,:,:);

x = XX2inner(:);
y = YY2inner(:);
z = ZZ2inner(:);

xyz = [x,y,z];

index = randperm(51000);
xyz = xyz(index,:);

save('TrainXYZ.mat', 'xyz')









