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
% N1 is usually 1, since y is column vector

if N1==1
    % Make each column zero-mean
    Y = bsxfun( @minus, Y, mean( Y, 1) );
    y = bsxfun( @minus, y, mean( y, 1) );
    
    y=repmat(y,1,N2);
    % L2 normalize each column
    Y = bsxfun( @times, Y, 1./sqrt( sum( Y.^2, 1) ) );
    y = bsxfun( @times, y, 1./sqrt( sum( y.^2, 1) ) );
    
    % Take the dot product of the columns and then sum
    s=sum( Y.*y, 1);
    s=abs(s);
    % disp(C);
else
    s = zeros(N1,N2);
    
    for i=1:N1
        for j = 1 : N2
            s(i,j) = abs(corr2(Y(:,j),y(:,i)));
        end
    end
end