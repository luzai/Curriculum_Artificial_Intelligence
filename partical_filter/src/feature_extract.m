function y = feature_extract(img, rect, sz_I, feature_type)
    % This function extract feature of image in rect
    % Note, image in rect should first be resized to sz_I, use imresize. 
    % For feature_type = 'intensity', if img has 3 channles(RGB image),
    % it should first be converted to gray scale.
    
    % Input:
    % img: input image
    % rect: [x, y, width, height] the region to extract feature
    % sz_I: [width, height] the standard width and height, this is to
    % ensure the dimension of features are the same.
    % feature_type: the type of feature, the default value is 'intensity'
    
    if nargin == 3
        feature_type = 'intensity';
    end
    if size(img, 3) == 3
        img = double(rgb2gray(img));
    end
    x = max(rect(1), 1);
    y = max(rect(2), 1);
    width = min(size(img, 2) - x -1, rect(3));
    height = min(size(img, 1) - y - 1, rect(4));
    rect = [x, y, width, height];
    
    roi = img(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3));
    

    switch feature_type
        case 'intensity'
            roi = imresize(roi, [sz_I(2), sz_I(1)]);
            y = double(roi(:)) / 255;
        case 'HOG'
%             pts=detectSURFFeatures(roi,'MetricThreshold',500);
% %             size(pts,1)
%             pts=selectStrongest(pts,3);
%             features=extractFeatures(roi,pts);
%             y=features(:);
            roi = imresize(roi, [sz_I(2), sz_I(1)]);
            y = double(extractHOGFeatures(roi));            
            y = y';
        case 'LBP'
            roi = imresize(roi, [sz_I(2), sz_I(1)]);
            y = double(extractLBPFeatures(roi));            
            y = y';
    end
 
end