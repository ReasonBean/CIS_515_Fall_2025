function [lamb,mu,w,b] = SVMhard2(rho,u,v)
%
%   Runs hard margin SVM version 2
%
%   p green vectors u_1, ..., u_p in n x p array u
%   q red   vectors v_1, ..., v_q in n x q array v
%
%   First builds the matrices for the dual program
%
p = size(u,2); q = size(v,2); n = size(u,1);
[A,c,X,Pa,qa] = buildhardSVM2(u,v);
%
%  Runs quadratic solver
%
tolr = 10^(-10); tols = 10^(-10); iternum = 180000;
[lam,U,nr,ns,kk] = qsolve1(Pa, qa, A, c, rho, tolr, tols, iternum);
if kk > iternum
   fprintf('** qsolve did not converge. Problem not solvable ** \n')
end
lamb = lam(1:p,1);
mu = lam(p+1:p+q,1);

%%%%%%
%%% Solve for w and b here, as well as numsvl1 and numsvm1
%%% numsvl1 is the count for nonzero lambda and numsvm1 is the number of nonzero mu
%%%%%%
w = -X*lam
tol = 10^(-10)
Ilam = []
for i = 1:p
   if lamb(i) > tol
      Ilam(end+1) = i
   end
end
Imu = []
for i = 1:q
   if mu(i) > tol
      Imu(end+1) = i
   end
end
numsvl1 = length(Ilam)
numsvm1 = length(Imu)
avg_l = zeros(2, 1)
for i = 1:numsvl1
   avg_l = avg_l + u(:, Ilam(i))
end
avg_l = avg_l / numsvl1
avg_m = zeros(2, 1)
for i = 1:numsvm1
   avg_m = avg_m + v(:, Imu(i))
end
avg_m = avg_m / numsvm1
b = dot(transpose(w), (avg_l + avg_m) / 2)

% Some additional error checking
nw = sqrt(w'*w);   % norm of w
delta = 1/nw;
if   delta < 10^(-9)
     fprintf('** Warning, delta too small, program does not converge ** \n')
end

if n == 2
   [ll,mm] = showdata(u,v);
   if numsvl1 > 0 && numsvm1 > 0
      showSVMs2(w,b,1,ll,mm,nw)
   end
end

