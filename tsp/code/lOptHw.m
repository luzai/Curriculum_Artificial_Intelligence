ccc
addpath(genpath('..\'));
all_pnt=load('res_latlon.txt');
figure,

lat=all_pnt(:,1);
lon=all_pnt(:,2);
lat_new=[];
lon_new=[];

for i=1:length(lat)
    longitude=lon(i);
    latitude=lat(i);
%     if ~(longitude<120.13 || longitude>120.21)
%         if ~(latitude<30.26 || latitude>30.3)
            lat_new(end+1)=lat(i);
            lon_new(end+1)=lon(i);
%             continue;
%         end
%     end
    
end
lat=lat_new;
lon=lon_new;

plot(lon,lat,'MarkerSize',8,'Marker','>',...
    'LineWidth',2,...
    'Color',[0 0 1])

plot_google_map
set(gcf,'Color','White');
legend('21_landmark')

% create xlabel
xlabel('Longitude (^o)');

% create ylabel
ylabel('Latitude (^o)');
title('GoogleMap JiangZheHu Range')
export_fig .\l-opt\GMap_Jzh.pdf
