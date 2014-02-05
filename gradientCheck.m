function gradientCheck
load('model.mat');
index = 1;
range = param(index)-1:0.01:param(index)+1;
param2 = param;
for i = 1:size(range,2);
    param2(index) = range(i);
    [f(i), g] = EigenGPNS_negLogLik(param2, x, y, M);
    df(i) = g(index);
end
for i = 2:size(range,2)-1;
    fdf(i-1) = (f(i+1)-f(i-1))/(range(i+1)-range(i-1));
end
subplot(2,1,1);
plot(range, f);
hold on;
plot(param(index), min(f), 'xr');
hold off;

subplot(2,1,2);
plot(range, df);
hold on;
plot(range(1,2:size(range,2)-1), fdf, 'r');
hold off;
end