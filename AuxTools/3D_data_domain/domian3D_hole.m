clear all
clc;
[x,y,z]=sphere(100);
hold on

% i=1;
% K=3;
for c11 = -0.8:0.4:0.8
    for c12 = -0.8:0.4:0.8
        for c13 = -0.8:0.4:0.8
            surf(0.075*x-c11,0.075*y-c12, 0.075*z-c13,'LineStyle','none','FaceColor', 'r');
        end
    end
end
hold on

for c1 = -0.8:0.8:0.8
    for c2 = -0.8:0.8:0.8
        for c3 = -0.8:0.8:0.8
            surf(0.125*x-c1,0.125*y-c2, 0.125*z-c3,'LineStyle','none','FaceColor', 'b');
        end
    end
end
hold on

camlight('headlight')