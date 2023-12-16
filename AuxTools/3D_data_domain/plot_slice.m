clear all
close all
clc

num2radius = 20;
num2theta = 60;

XX_Small1 = zeros(num2radius, num2theta);
YY_Small1 = zeros(num2radius, num2theta);

XX_Small2 = zeros(num2radius, num2theta);
YY_Small2 = zeros(num2radius, num2theta);

XX_Small3 = zeros(num2radius, num2theta);
YY_Small3 = zeros(num2radius, num2theta);

XX_Small4 = zeros(num2radius, num2theta);
YY_Small4 = zeros(num2radius, num2theta);

rho2small = 0.09;
rhos_Small = linspace(0.0,rho2small,num2radius);
thetas_Small = linspace(0.0,2*pi,num2theta);

for i=1:num2radius
    for j=1:num2theta
        XX_Small1(i,j) = rhos_Small(i).*cos(thetas_Small(j))+0.25;
        YY_Small1(i,j) = rhos_Small(i).*sin(thetas_Small(j))+0.25;
        
        XX_Small2(i,j) = rhos_Small(i).*cos(thetas_Small(j))+0.75;
        YY_Small2(i,j) = rhos_Small(i).*sin(thetas_Small(j))+0.75;
        
        XX_Small3(i,j) = rhos_Small(i).*cos(thetas_Small(j))+0.25;
        YY_Small3(i,j) = rhos_Small(i).*sin(thetas_Small(j))+0.75;
        
        XX_Small4(i,j) = rhos_Small(i).*cos(thetas_Small(j))+0.75;
        YY_Small4(i,j) = rhos_Small(i).*sin(thetas_Small(j))+0.25;
    end
end

ZZ_Small1 = ones(num2radius, num2theta)*0.5;
figure('name','inner_semisphere')
surf(XX_Small1,YY_Small1,ZZ_Small1, 'FaceColor', 'b', 'EdgeColor', 'none')
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$z$', 'Fontsize', 18, 'Interpreter', 'latex')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on

ZZ_Small2 = ones(num2radius, num2theta)*0.5;
surf(XX_Small2,YY_Small2,ZZ_Small2, 'FaceColor', 'b', 'EdgeColor', 'none')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on

ZZ_Small3 = ones(num2radius, num2theta)*0.5;
surf(XX_Small3,YY_Small3,ZZ_Small3, 'FaceColor', 'b', 'EdgeColor', 'none')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on

ZZ_Small4 = ones(num2radius, num2theta)*0.5;
surf(XX_Small4,YY_Small4,ZZ_Small4, 'FaceColor', 'b', 'EdgeColor', 'none')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on


XX_Big = zeros(num2radius, num2theta);
YY_Big = zeros(num2radius, num2theta);

rho = 0.16;
rhos = linspace(0.0,rho,num2radius);
thetas = linspace(0.0,2*pi,num2theta);

for i=1:num2radius
    for j=1:num2theta
        XX_Big(i,j) = rhos(i).*cos(thetas(j))+0.5;
        YY_Big(i,j) = rhos(i).*sin(thetas(j))+0.5;
    end
end

ZZ_Big = ones(num2radius, num2theta)*0.5;
surf(XX_Big,YY_Big,ZZ_Big, 'FaceColor', 'c', 'EdgeColor', 'none')
hold on
