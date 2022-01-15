%全局匹配，原方法
%ransac粗匹配要比直接用myicp好一些
clear;clc;

laser_map = pcread('0.ply');

tform_init = affine3d([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]);

robot_tf{1} = tform_init;

for i = 1:1:9
    
    disp(i);
    
    % read
    str = [num2str(i) , '.ply'];  
    curr_ply = pcread(str);

    % icp
    %[tform_init, curr_ply] = pcregistericp(curr_ply, laser_map, 'Metric','pointToPoint', 'InitialTransform', tform_init, 'MaxIterations', 100, 'Tolerance', [0.01, 0.001]);
    %[tform_init, curr_ply] = myicp(curr_ply, laser_map,"PointToLine","kdtree");
    [tform_init,curr_ply] = my_ransac_icp(curr_ply, laser_map,"relative");
    % robot_tf{i+1} = tform_init;
    
    % merge
    laser_map = pcmerge(laser_map, curr_ply, 0.01);
   
end

figure;
pcshow(laser_map, 'MarkerSize', 20);