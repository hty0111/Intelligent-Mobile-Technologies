function [goal] = simple_target_error_goal(curr_ply, laser_map ,goal_type)
    %误差计算函数，仅仅对图像进行简单匹配，即对目标点找出最邻近点，然后计算误差
    %goal_type为目标函数种类 "PointToPoint"--点到点，"PointToLine"--点到线
    P_raw = laser_map.Location';% 原地图
    P_prime_raw = curr_ply.Location';% 新地图
    max_count = 180; % 最多匹配180个点
    dmax = 25;%匹配点最大距离
    count=0;
    P=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%最近匹配点
    P2=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%次近匹配点
    P_prime=zeros(size(P_prime_raw,1),size(P_prime_raw,2));%目标点
    for i = 1:length(P_prime_raw(1, :))
        p_t = P_prime_raw(:, i);
%         if max(p_t) > 40
%             continue
%         end
        [indices,dists] = findNearestNeighbors(laser_map,p_t',2);%kd树近邻搜索出最近的两个点
        %返回的距离值
        mindist=dists(1);%最近点距离
        min2dist=dists(2);%次近点距离
        
        
        %返回的索引
        minidx=indices(1);%最近点索引
        min2idx=indices(2);%次近点索引
%         %如果距离过远，则认为由于一些因素不存在匹配点，那么舍弃
%         if mindist > dmax || min2dist > dmax
%             continue
%         end
% 
%         if max(P_raw(:, minidx)) > 40 || max(P_raw(:, min2idx)) > 40
%             continue
%         end
        
        P_prime(:,i) = p_t;%目标点
        P(:,i) = P_raw(:, minidx);%最近匹配点
        P2(:,i) = P_raw(:, min2idx);%次近匹配点

%         %把将要比对的点保存
%         count = count + 1;
% 
%         if count > max_count
%             break
%         end
    end
    %point to point
    PPd=zeros(1,size(P_prime,2));%点到点误差
    for i = 1:length(P_prime(1, :))
        PPd(i) = norm(P(:,i) - P_prime(:,i));%求距离向量的模
    end
    pp_goal = sum(PPd.^2)/2;
    %point to line
    PLd=zeros(1,size(P_prime,2));%点到线误差
    for i = 1:length(P_prime(1, :))
        Q2 = P2(:,i);
        Q1 = P(:,i);
        Pt = P_prime(:,i);
        PLd(i) = norm(cross(Q2-Q1,Pt-Q1))/norm(Q2-Q1);%求目标点到最近两个点的距离
    end
    PLd(isnan(PLd)) = 0;%将NaN的值替换为0
    pl_goal = sum(PLd.^2)/2;
    if strcmp(goal_type,"PointToPoint")
       goal = pp_goal;
    elseif strcmp(goal_type,"PointToLine")
        goal = pl_goal;
    end
    goal =  goal/length(P_prime_raw(1, :));
end