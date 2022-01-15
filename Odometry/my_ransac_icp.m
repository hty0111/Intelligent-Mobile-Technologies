function [tform,test_ply] = my_ransac_icp(curr_ply, laser_map,match_mode)
    %ransac随机采样方法的icp匹配函数，偏向于粗匹配
    %match_mode为匹配方式：
    %"original"--最原始的计算距离匹配（点云接近时效果好）
    %"relative"--相对位置查找匹配（点云差异较大时效果好）
    P_source = laser_map.Location';% 原地图
    P_target = curr_ply.Location';% 新地图
    %转移矩阵
    T_save=eye(4,4);
    %R旋转矩阵，t位移向量
    R_save=T_save(1:3,1:3);
    t_save=T_save(1:3,4);
    %点云中点的总个数
    len = length(P_target(1,:));
    %采样点个数
    nsam=3;
    %采样迭代最多次数
    max_iter=500;
    iter=0;
    %采样点匹配最小误差
    error_min=inf;
    %采样点最近距离
    min_between=1e-2;
    while true
        points = randperm(len,nsam);
        %预分配空间
        p=zeros(3,nsam);%目标点云采样点向量
        q=zeros(3,nsam);%初始点云采样点向量
        dpq=zeros(1,nsam);%采样点之间距离向量
        ind=zeros(1,nsam);%初始点云采样点序号向量
        if strcmp(match_mode,"original")
            %按距离匹配查找
            for i=1:nsam
                p(:,i) = P_target(:,points(i));
                dist_all = sum((P_source - p(:,i)).^2, 1); %求出新地图所有点到原地图一点的距离
                [dist,ind(i)] = min(dist_all); %求出最近的那一个点
                q(:,i) = P_source(:,ind(i));
                dpq(i)=dist;
            end
        elseif strcmp(match_mode,"relative")
            %相对距离查找法
            p(:,1) = P_target(:,points(1));%随机选取的第一个点
            ind(1) = randperm(len,1);%第一个匹配点也是随机选取
            q(:,1) = P_source(:,ind(1));%第一个匹配点坐标
            dpq(1) = norm(p(:,1)-q(:,1));%第一组目标点与匹配点之间的距离
            
            p(:,2) = P_target(:,points(2));%随机选取的第二个点
            pd12 = norm(p(:,1)-p(:,2));%p1与p2的距离
            dist_q1 = sqrt(sum((P_source - q(:,1)).^2, 1));%寻找所有q点到q1点的距离
            [~,ind(2)] = min(abs(dist_q1-pd12));%希望q1与q2的距离与p1与p2的距离最为接近
            q(:,2) = P_source(:,ind(2));%第二个匹配点坐标
            dpq(2) = norm(p(:,2)-q(:,2));%第二组目标点与匹配点之间的距离
            
            p(:,3) = P_target(:,points(3));%随机选取的第三个点
            pd32 = norm(p(:,3)-p(:,2));%p3与p2的距离
            pd31 = norm(p(:,3)-p(:,1));%p3与p1的距离
            dist_q2 = sqrt(sum((P_source - q(:,2)).^2, 1));%寻找所有q点到q1点的距离
            dist_q1 = sqrt(sum((P_source - q(:,1)).^2, 1));%寻找所有q点到q1点的距离
            %希望q3与q2、p3与p2的距离和与q3与q1、p3与p1的距离差之和最小
            [~,ind(3)] = min(abs(dist_q1-pd31)+abs(dist_q2-pd32));
            q(:,3) = P_source(:,ind(3));%第三个匹配点坐标
            dpq(3) = norm(p(:,3)-q(:,3));%第三组目标点与匹配点之间的距离
        end
        p_between=[norm(p(:,1)-p(:,2)),norm(p(:,1)-p(:,3)),norm(p(:,2)-p(:,3))];%p1,p2,p3之间的相互距离
        q_between=[norm(q(:,1)-q(:,2)),norm(q(:,1)-q(:,3)),norm(q(:,2)-q(:,3))];%q1,q2,q3之间的相互距离
        %采样点不能太近
        if find(p_between<min_between)>0
            continue
        end
        if find(q_between<min_between)>0
            continue
        end
        p_mean = mean(q, 2);
        p_prime_mean = mean(p, 2);
        Q = q - p_mean;
        Q_prime = p - p_prime_mean;
        W = Q_prime * Q';
        [U, ~, V] = svd(W);
        R = V * U'; %新地图到旧地图的旋转矩阵
        t = p_mean - R * p_prime_mean;
        P_test = R * P_target + t;
        test_ply = pointCloud(P_test');%计算出的新点云
        
        goal = simple_target_error_goal(test_ply,laser_map,"PointToPoint");%简单计算点云误差
        %找到最小误差采样点
        if error_min>goal
            error_min=goal;
            R_save=R;
            t_save=t;
        end
        %迭代足够次数下退出
        iter = iter+1;
        if iter>=max_iter
            break
        end
    end
    %计算最终点云结果
    P_test = R_save * P_target + t_save;
    test_ply = pointCloud(P_test');
    %计算最终目标函数值
    goal = simple_target_error_goal(test_ply,laser_map,"PointToPoint");
    disp(['ransac目标函数值为',num2str(goal*len)]);
    %保存转移矩阵
    T_save(1:3,1:3)=R_save;
    T_save(1:3,4)=t_save;
    tform = affine3d(T_save');
    %P_target = P_test;
    %pcshowpair(laser_map,test_ply, 'MarkerSize', 50);
end
