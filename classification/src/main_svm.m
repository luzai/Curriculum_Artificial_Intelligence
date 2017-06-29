% main_svm
clear;
try
%     !rm *.mat
%     !del *.mat
catch
    disp('remove fail')
end

addpath('../data/');
tic
%% data
[train_x,train_y]=cache_data('train',785);
[test_x,test_y]=cache_data('test');

disp('data ok')
%% training
classes=unique(train_y);
try
    load('final.mat')
    disp('model ok')
catch
    SVMModels=cell(10,1);
    
    parfor ind=1:10
        tic
        indx=train_y==classes(ind);
        %         train_t = 2*((train_y==ind-1) -0.5 );
        %         train_t=train_y==ind-1;
        %         assert(length(unique(train_t))==2);
        c=cvpartition(length(train_x),'KFold',10);
        fitfun=@(z)( fitcsvm(train_x,indx, ...
            'ClassNames',[false true],...
            'CVPartition',c,...
            'Standardize',true,...
            'KernelFunction','RBF',...
            'CacheSize','maximal',...
            'BoxConstraint',exp(z(1)),...
            'KernelScale','auto' ...
            )) ;
        minfun=@(z) (kfoldLoss(fitfun(z)));
        disp('start search')
        opts = optimset('TolX',5e-4,'TolFun',5e-4);
        m = 1;
        fval = zeros(m,1);
        z = zeros(1,1);
        for j = 1:m;
            disp('min and loss is')
            [searchmin fval(j)] = fminsearch(minfun,1+0.1*randn(1,1),opts);
            disp(exp(searchmin));disp(fval(j));
            z(j,:) = exp(searchmin);
        end
        
        z = z(fval == min(fval),:);
        SVMModel =( fitcsvm(train_x,indx, ...
            'ClassNames',[false true],...
            'Standardize',true,...
            'KernelFunction','RBF',...
            'KernelScale','auto',...
            'CacheSize','maximal',...
            'BoxConstraint',exp(z(1)))) ;
        
        
        
        SVMModels{ind}=SVMModel;
        
        disp('done!');disp(ind)
        toc
    end
    save('final.mat','SVMModels')
end


%% predict for test data
disp('start predict')
scores=zeros(length(test_x),numel(classes));
disp('len of test_x');disp(length(test_x));
for ind=1:numel(classes)
    [~,score]=predict(SVMModels{ind},test_x);
    scores(:,ind)=score(:,2);
end
[~,maxScore]=max(scores,[],2);
maxScore=maxScore-1;
ppp = maxScore==test_y;
% figure;hist(test_y,0:10);
% figure;hist(finalResult,0:10);
disp(['accuracy = ',num2str(sum(ppp)/10000*100),'%'])
toc