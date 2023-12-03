clc;
clear all
close all

x = linspace(-3,3,1000);
t0 = tanh(x);

figure('name','tanh')
plot(x,t0, 'b-','linewidth',2)
ylim([-1.1,1.1])
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

t1 = 1-tanh(x).*tanh(x);
figure('name','1st deri')
plot(x,t1, 'b-','linewidth',2)
ylim([0,1.1])
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on

t2 = 2*tanh(x).*tanh(x).*tanh(x) - 2*tanh(x);
figure('name','2nd deri')
plot(x,t2, 'b-','linewidth',2)
ylim([-1,1.1])
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 15);
set(gcf, 'Renderer', 'zbuffer');
hold on