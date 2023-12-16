clc;
clear all
close all

data2xyz = load('testXYZ.mat');
xyz = data2xyz.XYZ;
Solu = exp(-0.25*0.5)*sin(xyz(:,1)*pi).*sin(xyz(:,2)*pi)*sin(0.5*pi);
cp = Solu;
figure('name','domain')
scatter3(xyz(:,1),xyz(:,2),Solu,50, cp, '.')
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