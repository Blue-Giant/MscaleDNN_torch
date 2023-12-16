function h = drawSphere(r, centerx, centery, centerz, N)
if nargin == 5
    [x,y,z] = sphere(N);
else
    [x,y,z] = sphere(50);
end

h = surf(r*x+centerx, r*y+centery, r*z+centerz,'LineStyle','none','FaceColor', 'b');
h.EdgeColor = rand(1,3);
h.FaceColor = h.EdgeColor;