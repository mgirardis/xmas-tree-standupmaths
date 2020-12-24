clearvars
close all

c = importdata('xmastree2020\coords.txt',',');

r = cell2mat(cellfun(@(cc)str2double(strsplit(cc(2:(end-1)),',')),c,'UniformOutput',false));

plot3(r(:,1),r(:,2),r(:,3),'o')
daspect([1,1,1]);
axis square