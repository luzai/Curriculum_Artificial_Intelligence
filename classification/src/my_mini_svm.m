% my_mini_svm
clear;ccc
addpath('../data/');
%% data
clear;

addpath('../data/');
train_x = loadMNISTImages('train-images-idx3-ubyte');%768*60000
train_y = loadMNISTLabels('train-labels-idx1-ubyte');%60000*1

train_x = (train_x>0) + 0;train_x=train_x';%60000*768
train_x = cache_x(train_x,'train');

limits=600;
train_x=train_x(1:limits,:);
train_y=train_y(1:limits,:);

test_x = loadMNISTImages('t10k-images-idx3-ubyte');%768*10000
test_y = loadMNISTLabels('t10k-labels-idx1-ubyte');%10000*1
test_x = (test_x>0) + 0;test_x=test_x';
test_x=cache_x(test_x,'test');


tic

cls=0;
x=train_x;
y=train_y;
y(train_y==cls)=1;
y(train_y~=cls)=-1;
y=y';

[n,dim] = size(x);
C = 10;
plable = find(y==1);
nlable = find(y~=1);
plen = length(plable);
nlen = length(nlable);

options = optimset;    % Options
options.LargeScale = 'off';
options.Display = 'off';
%     H = (y'*y).*(x*x');
H = (y'*y).*svm_kernel(x,x);
%     f = -ones(n,1);
f = cat(1,zeros(plen,1),-ones(nlen,1));
A = [];
b = [];
Aeq = y;
beq = 0;
lb = zeros(n,1);
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
epsilon = 1e-8;
% display(a);
sv_label = find(abs(a)>epsilon);  %0<a<a(max)
a = a(sv_label);
Xsv = x(sv_label,:);
Ysv = y(sv_label);
svnum = length(sv_label);
model.a = a;
model.Xsv = Xsv;
model.Ysv = Ysv;

model.svnum = svnum;
num = length(Ysv);

W = zeros(1,dim);
for i = 1:num
    W = W+ a(i,1)*Ysv(i)*Xsv(i,:);
end
model.W = W;
%% calc b
py_label = find(Ysv==1);
pa = a(py_label);
pXsv = Xsv(py_label,:);
pYsv = Ysv(py_label);
pnum = length(py_label);
tmp = a'.*Ysv*svm_kernel(Xsv,pXsv);
b = -mean(tmp);
model.b = b;
%% test
parfor j = 1:length(test_y)
    test_j=test_x (j,:);
    tmp=0 ;
    for i = 1: svnum
        tmp=tmp+a(i)*Ysv(i)*svm_kernel(Xsv(i),test_j);
    end
    tmp=tmp+b;
    if tmp>0
        tmp=1;
    else
        tmp=-1;
    end
    res(j)=tmp;
end

res
test_y






