clear all
clc;
[x,y,z]=sphere(100);
axis equal
hold on

for c11 = 0.1:0.4:0.9
    for c12 = 0.1:0.4:0.9
        for c13 = 0.1:0.4:0.9
            surf(0.04*x-c11,0.04*y-c12, 0.04*z-c13,'LineStyle','none','FaceColor', 'r');
        end
    end
end
hold on

for c1 = 0.3:0.35:0.8
    for c2 = 0.3:0.35:0.8
        for c3 = 0.3:0.35:0.8
            surf(0.09*x-c1,0.09*y-c2, 0.09*z-c3,'LineStyle','none','FaceColor', 'b');
        end
    end
end
hold on
set(gca,'xtick',[])
set(gca,'ytick',[])
set(gca,'ztick',[])
axis on
grid on
box on
camlight('headlight')