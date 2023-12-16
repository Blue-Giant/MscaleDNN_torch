clear all
clc
close all
% entorid3D = rand(10,3, 0.1,0.9);

% for c11 = 0.05:0.3:0.95
%     for c12 = 0.05:0.3:0.95
%         for c13 = 0.05:0.3:0.95
%             [x1,y1,z1]  = ellipsoid(c11, c12, c13, 0.05, 0.05, 0.05,100);
%             surf(x1,y1,z1,'LineStyle','none','FaceColor', 'r') %画出来球
%             axis equal %保证各个维度的长短一致
%             hold on
%         end
%     end
% end
% hold on

% Big spere
[xb,yb,zb]  = ellipsoid(0.5, 0.5, 0.5, 0.16, 0.16, 0.16,100);
surf(xb,yb,zb,'LineStyle','none','FaceColor', 'c') %画出来球
axis equal %保证各个维度的长短一致
hold on

for c21 = 0.25:0.5:0.9
    for c22 = 0.25:0.5:0.9
        c23 = 0.5;
        [x2,y2,z2]  = ellipsoid(c21, c22, c23, 0.09, 0.09, 0.09,100);
        surf(x2,y2,z2,'LineStyle','none','FaceColor', 'b') %画出来球
        axis equal %保证各个维度的长短一致
    end
end
hold on

for c33 = 0.25:0.5:0.9
    for c32 = 0.25:0.5:0.9
        c31 = 0.5;
        [x3,y3,z3]  = ellipsoid(c31, c32, c33, 0.09, 0.09, 0.09,100);
        surf(x3,y3,z3,'LineStyle','none','FaceColor', 'r') %画出来球
        axis equal %保证各个维度的长短一致
    end
end
hold on
xlabel('$x$', 'Fontsize', 18, 'Interpreter', 'latex')
ylabel('$y$', 'Fontsize', 18, 'Interpreter', 'latex')
zlabel('$z$', 'Fontsize', 18, 'Interpreter', 'latex')
set(gca, 'XMinortick', 'off', 'YMinorTick', 'off', 'Fontsize', 14);
set(gcf, 'Renderer', 'zbuffer');
hold on

axis([0 1 0 1 0 1]);
camlight('headlight')




