function [dx, dy, Bx, By] = interpnatxy(x,y)
% This version uses the natural end condition
% Uses Matlab \ to solve linear systems
% Input points: two column vectors of x and y coordinates of dim N+1
%
%  This version uses x_0, x_1, ..., x_{N-1}, x_N to compute the Bezier
%  points and the subdivision version of the de Casteljau algorithm
%  to plot the Bezier segments (bspline2b and drawbezier_dc)
%
%  This version outputs the x and y coordinates dx and dy of the de Boor control
%  points d_{-1}, d_0, d_1, ..., d_{N+1} as column vectors
%  and the x and y coordinates of the Bezier control polygons
%  Bx and By
%

 %%% COMPUTE dx, dy, Bx, By HERE %%%

% Plots the spline
Nx = size(dx,1)-1;
fprintf('Nx = %d \n', Nx)
nn = 6; % subdivision level
drawb = true;
% hold on
[Bx, By] = bspline2b(dx,dy,Nx,nn,true);
hold on
plot(x,y,'b+'); % Plot x's as blue +
hold off;
end
