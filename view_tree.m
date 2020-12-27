clearvars
close all

c = importdata('xmastree2020\coords.txt',',');

r_LEDs = cell2mat(cellfun(@(cc)str2double(strsplit(cc(2:(end-1)),',')),c,'UniformOutput',false));

plot3(r_LEDs(:,1),r_LEDs(:,2),r_LEDs(:,3),'o')
axis square
daspect([1,1,1]);
grid on

% cone height above the z=0 plane
z0 = max(r_LEDs(:,3));

% cone total height
h = z0 + abs(min(r_LEDs(:,3)));

% cone base radius
r = (max(max(r_LEDs(:,2)),max(r_LEDs(:,1))) + abs(min(min(r_LEDs(:,2)),min(r_LEDs(:,1)))))/2;

% cone opening radius (defined by wolfram https://mathworld.wolfram.com/Cone.html )
c = r/h;

[x_grid,y_grid] = meshgrid(linspace(-r,r,100),linspace(-r,r,100));

z_cone = @(x,y,z0,c,s) z0+s.*sqrt((x.^2+y.^2)./(c.^2)); % s is the concavity of the cone: -1 turned down, +1 turned up
cone_r_sqr = @(z,z0,c) (c.*(z-z0)).^2;

hold on
spl = surf(x_grid,y_grid,z_cone(x_grid,y_grid,z0,c,-1));
% spl = surf(x_grid,y_grid,z_cone(x_grid,y_grid,z0+50,c+0.1,-1));
spl.EdgeColor = 'none';
spl.FaceAlpha=0.3;
axis square
daspect([1,1,1]);
grid on

outside_cone = (r_LEDs(:,1).^2+r_LEDs(:,2).^2) > cone_r_sqr(r_LEDs(:,3),z0,c);
plot3(r_LEDs(outside_cone,1),r_LEDs(outside_cone,2),r_LEDs(outside_cone,3),'o','MarkerFaceColor','k')
axis square
daspect([1,1,1]);
grid on