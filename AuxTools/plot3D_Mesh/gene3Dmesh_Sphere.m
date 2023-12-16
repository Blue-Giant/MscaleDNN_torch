clc;
clear all
close all

num2radius = 20;
num2theta = 60;
num2gamma = 50;

XX = zeros(num2radius,num2theta, num2gamma);
YY = zeros(num2radius,num2theta, num2gamma);
ZZ = zeros(num2radius,num2theta, num2gamma);

r = 2;
rho = linspace(1.0,r,num2radius);
theta = linspace(0.0,2*pi,num2theta);
gamma = linspace(0.0,pi,num2gamma);

for i=1:num2radius
    for j=1:num2theta
        XX(i,j,1:end) = rho(i)*sin(theta(j)).*cos(gamma);
        YY(i,j,1:end) = rho(i)*sin(theta(j)).*sin(gamma);
        ZZ(i,j,1:end) = rho(i)*cos(theta(j));
    end
end

XX2inner = XX(2:num2radius-1,:,:);
YY2inner = YY(2:num2radius-1,:,:);
ZZ2inner = ZZ(2:num2radius-1,:,:);

XX2inner1 = XX(2:num2radius/2-1,:,:);
YY2inner1 = YY(2:num2radius/2-1,:,:);
ZZ2inner1 = ZZ(2:num2radius/2-1,:,:);

XX2inner2 = XX(num2radius/2+1:num2radius-1,:,:);
YY2inner2 = YY(num2radius/2+1:num2radius-1,:,:);
ZZ2inner2 = ZZ(num2radius/2+1:num2radius-1,:,:);

xin1 = XX2inner1(:);
yin1 = YY2inner1(:);
zin1 = ZZ2inner1(:);

xin2 = XX2inner2(:);
yin2 = YY2inner2(:);
zin2 = ZZ2inner2(:);

xin = [xin1;xin2];
yin = [yin1;yin2];
zin = [zin1;zin2];
xyz = [xin,yin,zin];
index = randperm(51000);
xyz = xyz(index,:);

save('TrainXYZ.mat', 'xyz')

XX2Boundary1 = reshape(XX(1,:,:),num2theta, num2gamma);
YY2Boundary1 = reshape(YY(1,:,:),num2theta, num2gamma);
ZZ2Boundary1 = reshape(ZZ(1,:,:),num2theta, num2gamma);

xb1 = XX2Boundary1(:);
yb1 = YY2Boundary1(:);
zb1 = ZZ2Boundary1(:);

xyzbd1 = [xb1,yb1,zb1];
save('TrainXYZbd1.mat', 'xyzbd1')

XX2Boundary2 = reshape(XX(num2radius,:,:),num2theta, num2gamma);
YY2Boundary2 = reshape(YY(num2radius,:,:),num2theta, num2gamma);
ZZ2Boundary2 = reshape(ZZ(num2radius,:,:),num2theta, num2gamma);

xb2 = XX2Boundary2(:);
yb2 = YY2Boundary2(:);
zb2 = ZZ2Boundary2(:);

xyzbd2 = [xb2,yb2,zb2];
save('TrainXYZbd2.mat', 'xyzbd2')

fb1 = sin(pi*xb1).*sin(pi*yb1).*sin(pi*zb1);
FB1 = reshape(fb1, num2theta, num2gamma);
figure('name','inner_sphere')
surf(XX2Boundary1, YY2Boundary1, ZZ2Boundary1, FB1)
hold on




