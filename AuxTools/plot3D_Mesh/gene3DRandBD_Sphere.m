clc;
clear all
close all

num2theta = 100;
num2gamma = 100;

XX_BD1 = zeros(num2theta, num2gamma);
YY_BD1 = zeros(num2theta, num2gamma);
ZZ_BD1 = zeros(num2theta, num2gamma);

XX_BD2 = zeros(num2theta, num2gamma);
YY_BD2 = zeros(num2theta, num2gamma);
ZZ_BD2 = zeros(num2theta, num2gamma);

r1 = 1.0;
r2 = 2.0;
theta = 0+2*pi*rand(num2theta, 1);
gamma = 0+pi*rand(num2gamma, 1);

for j=1:num2theta
    XX_BD1(j,1:end) = r1*sin(theta(j)).*cos(gamma);
    YY_BD1(j,1:end) = r1*sin(theta(j)).*sin(gamma);
    ZZ_BD1(j,1:end) = r1*cos(theta(j));

    XX_BD2(j,1:end) = r2*sin(theta(j)).*cos(gamma);
    YY_BD2(j,1:end) = r2*sin(theta(j)).*sin(gamma);
    ZZ_BD2(j,1:end) = r2*cos(theta(j));
end

xb1 = XX_BD1(:);
yb1 = YY_BD1(:);
zb1 = ZZ_BD1(:);

xyzbd1 = [xb1,yb1,zb1];
save('TrainXYZbd1.mat', 'xyzbd1')

xb2 = XX_BD2(:);
yb2 = YY_BD2(:);
zb2 = ZZ_BD2(:);

xyzbd2 = [xb2,yb2,zb2];
save('TrainXYZbd2.mat', 'xyzbd2')

fb1 = sin(pi*xb1).*sin(pi*yb1).*sin(pi*zb1);
FB1 = reshape(fb1, num2theta, num2gamma);
figure('name','inner_sphere')
% surf(XX_BD1, YY_BD1, ZZ_BD1, FB1)
scatter3(xb1, yb1, zb1, fb1)
hold on




