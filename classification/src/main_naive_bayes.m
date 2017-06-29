% main_naive_bayes
%% data
try
    ccc
catch
end
clear all;

addpath('../data/');
train_x = loadMNISTImages('train-images-idx3-ubyte');%768*60000
train_y = loadMNISTLabels('train-labels-idx1-ubyte');%60000*1

% train_x = (train_x>0) + 0;
train_x=train_x';

test_x = loadMNISTImages('t10k-images-idx3-ubyte');%768*10000
test_y = loadMNISTLabels('t10k-labels-idx1-ubyte');%10000*1
% test_x = (test_x>0) + 0;
test_x=test_x';

tic

classes=0:9;
nb_cls=10; % number of classes
nb_features=size(train_x,2); % independent variables
nb_tests=length(test_y); % test set
nb_trains=length(train_x);
method = 'multinormal'; % 'normal';''
%% Prior
for i=1:nb_cls
    py(i)=sum(double(train_y==classes(i)))/length(train_y);
end

%% training
% assume normal distribution and **independently**!
% parameters from training set
switch method
    case 'normal'
        for i=1:nb_cls
            xi=train_x((train_y==classes(i)),:);
            cnt(i,:)=mean(xi,1);
            assert(all(mean(xi,1)==sum(xi,1)/length(xi)));
            %     mu(i,:)=mean(xi,1);
            %     sigma(i,:)=std(xi,1);
        end
        
    case 'multinormal'
        for i=1:nb_cls
            groupI=(train_y==i-1);
            pw_i=sum(train_x(groupI,:),1);
            pw_i=(pw_i+1)/(sum(pw_i)+nb_trains);
            assert(all(pw_i~=0));
            pw(i,:)=pw_i;
        end
end

%% testing
switch method
    case 'normal'
        for j=1:nb_tests
            test_x_j=test_x(j,:);
            for i=1:nb_cls
                cls=classes(i);
                %        mu_i=mu(i,:);
                %        sigma_i=sigma(i,:);
                %        px_giveny_i=normcdf(ones(1,nb_features)*0.45,mu_i,sigma_i);
                px_giveny_i=cnt(i,:);
                px_giveny_i=1-test_x_j-px_giveny_i;
                px_giveny(i,:)=px_giveny_i;
            end
            py_givenx(j,:)=py.*prod(px_giveny,2)';
            %     px_giveny2=normcdf(ones(nb_cls,1)*test_x(j,:),mu,sigma);
            %     py_givenx2(j,:)=py.*prod(px_giveny2,2)';
        end
        
        
        % get predicted output for test set
        [~,pred_ind]=max(py_givenx,[],2);
        for i=1:length(pred_ind)
            pred(i,1)=classes(pred_ind(i));
        end
    case 'multinormal'
        logpw=log(pw);
        log_coef=gammaln(nb_features+1)-sum(gammaln(test_x+1),2);
        log_cond_pdf=bsxfun(@plus,test_x*logpw',log_coef);
        log_pdf=bsxfun(@plus,log_cond_pdf,log(py));
        [~,pred]=max(log_pdf,[],2);
        pred=pred-1;
end

% figure;histogram(test_y,0:10);
% figure;histogram(pred,0:10);
%% calc acc
conf=sum(pred==test_y)/length(pred);
disp(['accuracy = ',num2str(conf*100),'%'])

toc