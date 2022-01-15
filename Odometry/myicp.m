function [tform, curr_ply] = myicp(curr_ply, laser_map ,goal_type,match_mode)
    %goal_type为目标函数种类 "PointToPoint"--点到点，"PointToLine"--点到线
    %match_mode为匹配方式 "original"--最原始的计算距离匹配，"kdtree"--kd树邻近搜索匹配
    P_raw = laser_map.Location'; % 原地图，命名考虑到和PPT公式一致，化为3行×180列的形式，每列是一个点
    P_prime_raw = curr_ply.Location'; % 新地图，命名考虑到和PPT公式一致，化为点数3行×180列的形式
    max_count = 180; % 最多匹配180个点
    % ICP 线性代数迭代求解
    %转移矩阵
    T_save=eye(4,4);
    %R旋转矩阵，t位移向量
    R_save=T_save(1:3,1:3);
    t_save=T_save(1:3,4);
    %粗配准
    % ply_target=R_save*ply_target+T_save*ones(1,size(ply_target,2))
    %匹配点最大距离
    dmax = 25;
    %目标误差
    goal=inf;
    min_goal=0.5;
    %迭代次数
    iter=0;
    max_iter=200;
    
    while goal > min_goal
        count = 0;
        iter = iter+1;
        %分配空表空间
        P=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%最近匹配点
        P2=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%次近匹配点
        P_prime=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%目标点
        
        for i = 1:length(P_prime_raw(1, :)) %对于新地图所有点
            p_prime_raw = P_prime_raw(:, i);

            if max(p_prime_raw) > 40
                continue
            end
            
            if strcmp(match_mode,"original")
                
                dist = sum((P_raw - p_prime_raw).^2, 1); %求出新地图所有点到原地图一点的距离
                [mindist, minidx] = min(dist); %求出最近的那一个点
                
                dist(minidx) = max(dist);%剔除掉最近点
                [min2dist, min2idx] = min(dist);%求出第二近的点
                
            elseif strcmp(match_mode,"kdtree")
                
                [indices,dists] = findNearestNeighbors(laser_map,p_prime_raw',2);%kd树近邻搜索出最近的两个点
                %返回的距离值
                mindist=dists(1);%最近点距离
                min2dist=dists(2);%次近点距离
                %返回的索引
                minidx=indices(1);%最近点索引
                min2idx=indices(2);%次近点索引
                
            end

            %如果距离过远，则认为由于一些因素不存在匹配点，那么舍弃
            if mindist > dmax || min2dist > dmax
                continue
            end

            if max(P_raw(:, minidx)) > 40 || max(P_raw(:, min2idx)) > 40
                continue
            end

            %把将要比对的点保存
            count = count + 1;

            if count > max_count
                break
            end

            P_prime(:,i) = p_prime_raw;%目标点
            P(:,i) = P_raw(:, minidx);%最近匹配点
            P2(:,i) = P_raw(:, min2idx);%次近匹配点
        end
        
        %计算质心位置
        p_mean = mean(P, 2);
        p_prime_mean = mean(P_prime, 2);
        Q = P - p_mean;
        Q_prime = P_prime - p_prime_mean;
        
        W = Q_prime * Q'; % 3*点数 * 点数*3 = 3*3
        [U, ~, V] = svd(W);
        R = V * U'; %新地图到旧地图的旋转矩阵
        t = p_mean - R * p_prime_mean;
        % 保存本次变换后点云
        P_prime_raw = R * P_prime_raw + t;
        %计算优化目标函数
        P_target = R * P_prime + t;
        %计算优化目标函数
        %point to point
        PPd=zeros(1,size(P_prime,2));%点到点误差
        for i = 1:length(P_prime(1, :))
            PPd(i) = norm(P(:,i) - P_target(:,i));%求距离向量的模
        end
        pp_goal = sum(PPd.^2)/2;
        %point to line
        PLd=zeros(1,size(P_prime,2));%点到线误差
        for i = 1:length(P_prime(1, :))
            Q2 = P2(:,i);
            Q1 = P(:,i);
            Pt = P_target(:,i);
            PLd(i) = norm(cross(Q2-Q1,Pt-Q1))/norm(Q2-Q1);%求目标点到最近两个点的距离
        end
        PLd(isnan(PLd)) = 0;%将NaN的值替换为0
        pl_goal = sum(PLd.^2)/2;
        %选择最终目标函数
        if strcmp(goal_type,"PointToPoint")
           goal = pp_goal;
        elseif strcmp(goal_type,"PointToLine")
            goal = pl_goal;
        end
        %保存最开始地图到现在的变换
        R_save = R * R_save;
        t_save = R * t_save + t;
        %最大迭代次数后停止
        if iter>=max_iter
            break
        end
    end
    disp(['最终迭代次数为',num2str(iter)]);
    disp(['最终目标函数值为',num2str(goal)]);
    tform = rigid3d(R_save',t_save');
    curr_ply = pointCloud(P_prime_raw');
end

