function s = compute_similarity(Y, y)
% Compute similarity between y and Y
% Input:
% Y: a dxN matrix, d is the dimension of feature, N is number of particle
% Y contains features for each particle
% y: feature of the last tracked rect
% Ouptput:
% s: a vector indicates the similarity between each column of Y and y
[n_features,N1]=size(y);
[n_features,N2]=size(Y);

s = zeros(N1,N2);
simi_func=@(a,b) abs(corr2(a,b));
for i=1:N1
%     s(i,:)=bsxfun(simi_func,Y,y(:,i));
    for j = 1 : N2
        s(i,j) = abs(corr2(Y(:,j),y(:,i)));
    end
end
