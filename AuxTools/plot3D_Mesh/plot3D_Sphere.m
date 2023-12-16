clc;
clear all
close all

figure('name', 'standard sphere')
num2points = 20;
[x,y,z] = sphere(num2points);
surf(x,y,z)
hold on

figure('name','func_sphere')
[xf,yf,zf] = sphere(num2points);
f = sin(pi*xf).*sin(pi*yf).*sin(pi*zf);
surf(xf,yf,zf,f)
colorbar
hold on


