clear all
close all
clc

num2point = 100000;
xy_point = rand(num2point, 2);
interiork=1;
for ip = 1:num2point
    x_ip = xy_point(ip,1);
    y_ip = xy_point(ip,2);
    
    cx_b = x_ip-0.5;
    cy_b = y_ip-0.5;
    rcb = sqrt(cx_b^2+cy_b^2);
    
    cx_1 = x_ip-0.25;
    cy_1 = y_ip-0.25;
    rc1 = sqrt(cx_1^2+cy_1^2);
    
    cx_2 = x_ip-0.75;
    cy_2 = y_ip-0.75;
    rc2 = sqrt(cx_2^2+cy_2^2);
    
    cx_3 = x_ip-0.25;
    cy_3 = y_ip-0.75;
    rc3 = sqrt(cx_3^2+cy_3^2);
    
    cx_4 = x_ip-0.75;
    cy_4 = y_ip-0.25;
    rc4 = sqrt(cx_4^2+cy_4^2);
    if rcb>0.16 && rc1>0.09 && rc2>0.09 && rc3>0.09 && rc4>0.09
        irregularD(interiork, 1)=x_ip;
        irregularD(interiork, 2)=y_ip;
        interiork=interiork+1;
    end
end

% figure('name', 'fig')
% scatter(irregularD(1,:),irregularD(2,:),'r.')
% hold on

Solu = exp(-0.25*0.5)*sin(irregularD(:,1)*pi).*sin(irregularD(:,2)*pi)*sin(0.5*pi);
cp = Solu;
figure('name','domain')
scatter3(irregularD(:,1),irregularD(:,2),Solu,50, cp, '.')
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$u$', 'Fontsize', 18, 'Interpreter', 'latex')
shading interp
hold on
colorbar;
caxis([0 1])
hold on
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

Z = ones(interiork-1, 1)*0.5;
XYZ = [irregularD, Z];
save('testXYZ.mat','XYZ')
