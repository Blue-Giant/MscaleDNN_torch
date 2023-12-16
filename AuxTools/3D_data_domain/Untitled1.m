clear all
clc
close all
% entorid3D = rand(10,3, 0.1,0.9);

for c11 = 0.05:0.3:0.95
    for c12 = 0.05:0.3:0.95
        for c13 = 0.05:0.3:0.95
            [x1,y1,z1]  = ellipsoid(c11, c12, c13, 0.05, 0.05, 0.05,100);
            surf(x1,y1,z1,'LineStyle','none','FaceColor', 'r') %画出来球
            axis equal %保证各个维度的长短一致
            hold on
        end
    end
end
hold on

for c21 = 0.2:0.3:0.8
    for c22 = 0.2:0.3:0.8
        for c23 = 0.2:0.3:0.8
            [x2,y2,z2]  = ellipsoid(c21, c22, c23, 0.1, 0.1, 0.1,100);
            surf(x2,y2,z2,'LineStyle','none','FaceColor', 'b') %画出来球
            axis equal %保证各个维度的长短一致
            hold on
        end
    end
end
hold on

axis([0 1 0 1 0 1]);
camlight('headlight')