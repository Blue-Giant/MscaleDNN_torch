clc;
clear all
close all

num2radius = 20;
num2theta = 100;
num2gamma = 80;
% [rho0,theta0,gamma0] = meshgrid(linspace(0.5,2, num2radius),linspace(0,2*pi, num2theta),linspace(0,pi, num2gamma));

XX = zeros(num2radius,num2theta, num2gamma);
YY = zeros(num2radius,num2theta, num2gamma);
ZZ = zeros(num2radius,num2theta, num2gamma);

r = 2;
rho = linspace(0.1,1.0,num2radius);
theta = linspace(0.0,2*pi,num2theta);
gamma = linspace(0.0,pi,num2gamma);
% [theta,gamma] = meshgrid(linspace(0,2*pi, num2theta),linspace(0,pi, num2gamma));

for i=1:num2radius
    for j=1:num2theta
        XX(i,j,1:end) = rho(i)*sin(theta(j)).*cos(gamma);
        YY(i,j,1:end) = rho(i)*sin(theta(j)).*sin(gamma);
        ZZ(i,j,1:end) = rho(i)*cos(theta(j));
    end
end

XX2Exact = reshape(XX(num2radius/2,1:end,1:end), num2theta,num2gamma);
YY2Exact = reshape(YY(num2radius/2,1:end,1:end), num2theta,num2gamma);
ZZ2Exact = reshape(ZZ(num2radius/2,1:end,1:end), num2theta,num2gamma);
x = XX2Exact(:);
y = YY2Exact(:);
z = ZZ2Exact(:);
xyz = [x,y,z];
save('TestXYZ.mat','xyz')

disorder_index = randperm(num2theta*num2gamma);
XYZ_shufle = xyz(disorder_index, :);

save('Shufle_TestXYZ.mat','XYZ_shufle')
save('Shufle_Index.mat','disorder_index')




