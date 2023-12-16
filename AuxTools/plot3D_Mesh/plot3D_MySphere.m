clc;
clear all
close all

num2radius = 20;
num2theta = 60;
num2gamma = 50;
% [rho0,theta0,gamma0] = meshgrid(linspace(0.5,2, num2radius),linspace(0,2*pi, num2theta),linspace(0,pi, num2gamma));

XX = zeros(num2radius,num2theta, num2gamma);
YY = zeros(num2radius,num2theta, num2gamma);
ZZ = zeros(num2radius,num2theta, num2gamma);

r = 1.0;
rho = linspace(0.1,r,num2radius);
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

XX1 = reshape(XX(1,1:end,1:end), num2theta,num2gamma);
YY1 = reshape(YY(1,1:end,1:end), num2theta,num2gamma);
ZZ1 = reshape(ZZ(1,1:end,1:end), num2theta,num2gamma);
figure('name','inner_sphere')
surf(XX1,YY1,ZZ1)
hold on

XX2 = reshape(XX(num2radius,1:end,1:end), num2theta,num2gamma);
YY2 = reshape(YY(num2radius,1:end,1:end), num2theta,num2gamma);
ZZ2 = reshape(ZZ(num2radius,1:end,1:end), num2theta,num2gamma);
figure('name','out_sphere')
surf(XX2,YY2,ZZ2)
hold on

XX3 = reshape(XX(1,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);
YY3 = reshape(YY(1,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);
ZZ3 = reshape(ZZ(1,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);

XX4 = reshape(XX(num2radius,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);
YY4 = reshape(YY(num2radius,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);
ZZ4 = reshape(ZZ(num2radius,1:num2theta/2,1:num2gamma), num2theta/2,num2gamma);

figure('name','inner_semisphere')
surf(XX3,YY3,ZZ3)
colorbar
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$z$', 'Fontsize', 18, 'Interpreter', 'latex')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on
surf(XX4,YY3,ZZ4)
hold on


XX2Exact = reshape(XX(num2radius/2,1:end,1:end), num2theta,num2gamma);
YY2Exact = reshape(YY(num2radius/2,1:end,1:end), num2theta,num2gamma);
ZZ2Exact = reshape(ZZ(num2radius/2,1:end,1:end), num2theta,num2gamma);
x = XX2Exact(:);
y = YY2Exact(:);
z = ZZ2Exact(:);
xyz = [x,y,z];
save('TestXYZ.mat','xyz')

% FF = sin(pi*XX2Exact).*sin(pi*YY2Exact).*sin(pi*ZZ2Exact);
% figure('name','exact')
% surf(XX2Exact,YY2Exact, ZZ2Exact, FF);
% hold on
% xlabel('X')
% ylabel('Y')
% zlabel('Z')
% colorbar
% hold on
% light('Position',[-2,-2,2.5])
% hold on



