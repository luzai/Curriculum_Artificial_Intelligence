function [x,y]=cache_data(mode,limits)
if nargin==0
    ccc
    mode='train';
end
if strcmp(mode,'train')
    x_name='train-images-idx3-ubyte';
    y_name='train-labels-idx1-ubyte';
else
    x_name='t10k-images-idx3-ubyte';
    y_name='t10k-labels-idx1-ubyte';
end
x=loadMNISTImages(x_name);
y=loadMNISTLabels(y_name);
x=x';

if strcmp(mode,'train')
%     limits=785;%limits=7850;
    x=x(1:limits,:);
    y=y(1:limits,:);
end

try
    disp('use cache')
    load([mode '_mnist.mat'])
%      x=transform(x);
catch
    disp('nocache')
    x=transform(x);
    save([mode '_mnist.mat'],'x')
end


    function res=transform(x)
        parfor i = 1:length(x)
            x_i=reshape(x(i,:),28,28);
            %             x_i=(x_i>0)+0;
            %             x_i=lbp(x_i)
            x_i=extractHOGFeatures(x_i);
            %               x_i=extractLBPFeatures(x_i)
            %             x_i=reshape(x(i,:),1,28*28);
            %             disp(length(x_i))
            res(i,:)=x_i;
        end
    end


end