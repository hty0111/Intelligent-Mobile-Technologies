%逐桢匹配，针对于静匹配
%整体效果myicp比ransac好很多，且ransac不稳定
clear;clc;close all

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
Tx1=robot_tf{1,1}.T';
for i=2:10
    Tx1 = cat(3,Tx1,robot_tf{1,i}.T');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    [tform_init, curr_ply] = pcregistericp(curr_ply, laser_map, 'Metric','pointToPoint', 'InitialTransform', tform_init, 'MaxIterations', 100, 'Tolerance', [0.01, 0.001]);

    %[tform_init, curr_ply] = myicp(curr_ply, Q,"PointToLine","kdtree");
    %[tform_init,curr_ply] = my_ransac_icp(curr_ply, Q,"relative");
    robot_tf{i+1} = tform_init;
    
    % merge
    laser_map = pcmerge(laser_map, curr_ply, 0.01);
    Q=curr_ply;
end
figure;
pcshow(laser_map, 'MarkerSize', 20);


Tx2=robot_tf{1,1}.T';
for i=2:10
    Tx2 = cat(3,Tx2,robot_tf{1,i}.T');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
%Tx2为原方法  %Tx1为自定义方法
tran2=transl(Tx2);%轨迹
tr2=tr2rpy(Tx2);%位姿（欧拉角）
tr2=tr2(:,3);

tran1=transl(Tx1);
tr1=tr2rpy(Tx1);
tr1=tr1(:,3);

% tran1(:,1:2);
% tran2(:,1:2);
tran_err=0;%轨迹误差
tr_err=0;%位姿误差
len = length(tran1(:,1));%位姿个数
for i = 1:len
    tran_err=tran_err+norm(tran1(i,1:2)-tran2(i,1:2));%取轨迹途径点的欧式距离
    tr_err=tr_err+abs(tr1(i)-tr2(i));%取欧拉角的差值
end
%取均值
tran_err = tran_err/len;
tr_err = tr_err/len;
disp(['最终轨迹误差为',num2str(tran_err)]);
disp(['最终位姿(欧拉角)误差为',num2str(tr_err)]);
%轨迹对比可视化
figure
hold on
plot(tran1(:,1),tran1(:,2),"r");
plot(tran2(:,1),tran2(:,2),"b");
hold off
%位姿(欧拉角)对比可视化
figure
hold on;
plot(tr1,"r");
plot(tr2,"b");
hold off