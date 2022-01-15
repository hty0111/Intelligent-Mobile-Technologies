%逐桢匹配，针对于静匹配
%整体效果myicp比ransac好很多，且ransac不稳定
clc;close all

laser_map = pcread('0.ply');

tform_init = affine3d([1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]);

robot_tf=cell(1,10);
robot_tf{1} = tform_init;
Q=laser_map;
for i = 1:1:9
    
    disp(i);
    
    % read
    str = [num2str(i) , '.ply'];  
    curr_ply = pcread(str);
    
    % icp
    %[tform_init, curr_ply] = pcregistericp(curr_ply, laser_map, 'Metric','pointToPoint', 'InitialTransform', tform_init, 'MaxIterations', 100, 'Tolerance', [0.01, 0.001]);

    [tform_init, curr_ply] = myicp(curr_ply, Q,"PointToLine","kdtree");
    %[tform_init,curr_ply] = my_ransac_icp(curr_ply, Q,"relative");
    robot_tf{i+1} = tform_init;
    
    % merge
    laser_map = pcmerge(laser_map, curr_ply, 0.01);
    Q=curr_ply;
end
figure;
pcshow(laser_map, 'MarkerSize', 20);
%位姿轨迹可视化插值个数
internum = 10;
Ts=cell(1,9);
for i = 1:9
    Ts{i} = trinterp(robot_tf{1,i}.T',robot_tf{1,i+1}.T',(0:internum-1)'/(internum-1));
end
%将所有位姿并列在一起
Tx=cat(3,Ts{1},Ts{2});
for i = 3:9
    Tx=cat(3,Tx,Ts{i});
end
figure
%tranimate(Tx)
%生成gif
tranimate(Tx,'movie','route.gif');
% figure
% plot(transl(Tx));
% figure
% plot(tr2rpy(Tx));