function [] = ekf_localization()
 
% Homework for ekf localization
% Modified by YH on 09/09/2019, thanks to the original open source
% Any questions please contact: zjuyinhuan@gmail.com

    close all;
    clear;

    disp('EKF Start!')

    time = 0;
    global endTime; % [sec]
    endTime = 60;
    global dt;
    dt = 0.1; % [sec]

    removeStep = 5;

    nSteps = ceil((endTime - time)/dt);

    estimation.time=[];
    estimation.u=[];
    estimation.GPS=[];
    estimation.xOdom=[];
    estimation.xEkf=[];
    estimation.xTruth=[];

    % State Vector [x y yaw]'
    xEkf = [0 0 0]';
    PxEkf = eye(3);

    % Ground True State
    xTruth = xEkf;

    % Odometry Only
    xOdom = xTruth;

    % Observation vector [x y yaw]'
    z = [0 0 0]';

    % Simulation parameter
    global noiseQ
    noiseQ = diag([0.1 0 degreeToRadian(10)]).^2; %[Vx Vy yawrate]

    global noiseR
    noiseR = diag([0.5 0.5 degreeToRadian(5)]).^2;%[x y yaw]
    
    % Covariance Matrix for motion
    convQ = 0.05 .* noiseQ * noiseQ';
%     convQ = 0.01 .* eye(3);
%     convQ = 5 .* noiseQ.^2;
%     convQ = 0.01 .* noiseQ;

    % Covariance Matrix for observation
    convR = 0.5 .* noiseR * noiseR';
%     convR = 0.01 .* eye(3);
%     convR = 5 .* noiseR.^2;
%     convR = 0.05 .* noiseR;

    % Other Intial
    xPred = xEkf;
    

    % Main loop
    for i = 1 : nSteps
        time = time + dt;
        % Input
        u=robotControl(time);
        % Observation
        [z,xTruth,xOdom,u]=prepare(xTruth, xOdom, u);

        % ------ Kalman Filter --------
        % Predict   
        xPred =  doMotion(xEkf, u);         % 根据系统模型得到先验估计值
        JF = jacobF(xEkf, u);               % 计算雅克比矩阵JF
        PxEkf = JF * PxEkf * JF' + convQ;   % 预测误差的协方差矩阵
        
        % Update
        JH = jacobH(xPred);                             % 计算雅克比矩阵JH
        K = PxEkf * JH' * (JH*PxEkf*JH' + convR)^-1;    % 计算卡尔曼增益
        xEkf = doObservation(z,xPred,K);                % 根据测量值更新观测值
        PxEkf = (eye(3) - K*JH) * PxEkf;                % 更新误差的协方差矩阵
        % -----------------------------

        % Simulation estimation
        estimation.time=[estimation.time; time];
        estimation.xTruth=[estimation.xTruth; xTruth'];
        estimation.xOdom=[estimation.xOdom; xOdom'];
        estimation.xEkf=[estimation.xEkf;xEkf'];
        estimation.GPS=[estimation.GPS; z'];
        estimation.u=[estimation.u; u'];

        % Plot in real time
        % Animation (remove some flames)
        if rem(i,removeStep)==0
            %hold off;
            plot(estimation.GPS(:,1),estimation.GPS(:,2),'*m', 'MarkerSize', 5);hold on;
            plot(estimation.xOdom(:,1),estimation.xOdom(:,2),'.k', 'MarkerSize', 10);hold on;
            plot(estimation.xEkf(:,1),estimation.xEkf(:,2),'.r','MarkerSize', 10);hold on;
            plot(estimation.xTruth(:,1),estimation.xTruth(:,2),'.b', 'MarkerSize', 10);hold on;
            axis equal;
            grid on;
            drawnow;
            %movcount=movcount+1;
            %mov(movcount) = getframe(gcf);
        end 
    end
    close
    
    finalPlot(estimation);
 
end

% control
function u = robotControl(time)
    global endTime;

    T = 10; % sec
    Vx = 1.0; % m/s
    Vy = 0.2; % m/s
    yawrate = 5; % deg/s
    
    % half
    if time > (endTime/2)
        yawrate = -5;
    end
    
    u =[ Vx*(1-exp(-time/T)) Vy*(1-exp(-time/T)) degreeToRadian(yawrate)*(1-exp(-time/T))]';
    
end

% all observations for 
function [z, xTruth, xOdom, u] = prepare(xTruth, xOdom, u)
    global noiseQ;
    global noiseR;

    % Ground Truth
    xTruth = doMotion(xTruth, u);
    % add Motion Noises
    u = u + noiseQ * randn(3,1);
    % Odometry Only
    xOdom = doMotion(xOdom, u);
    % add Observation Noises
    z = xTruth + noiseR * randn(3,1);   % 测量值 GPS得到
end

% Motion Model
function x = doMotion(x, u)
    global dt;
    v = sqrt(u(1)^2 + u(2)^2);  w = u(3);   theta = x(3);
    x(1) = x(1) - v/w*sin(theta) + v/w*sin(theta+w*dt);
    x(2) = x(2) + v/w*cos(theta) - v/w*cos(theta+w*dt);
    x(3) = x(3) + dt * w;
%     x(1) = x(1) + dt * v * cos(x(3));
%     x(2) = x(2) + dt * v * sin(x(3));
%     x(3) = x(3) + dt * w;
end

% Jacobian of Motion Model
function jF = jacobF(X, u)
    global dt;
%     v = sqrt(u(1)^2 + u(2)^2);
%     w = u(3);
%     syms x y theta
%     f = [x - v/w*sin(theta) + v/w*sin(theta+w*dt); 
%         y + v/w*cos(theta) - v/w*cos(theta+w*dt);
%         theta + dt * w];
%     J = jacobian(f, [x y theta]);
%     jF = subs(J, [x y theta], X);

    v = sqrt(u(1)^2 + u(2)^2);  w = u(3);   theta = X(3);
    jF=[1 0 v/w * (-cos(theta)+cos(dt*w+theta)); 
        0 1 1/w * (cos(theta)-cos(dt*w+theta));
        0 0 1];
end

%Observation Model
function x = doObservation(z, xPred, K)
    x = xPred + K * (z-xPred);
 end

%Jacobian of Observation Model
function jH = jacobH(x)
    jH = eye(3);
end

% finally plot the results
function []=finalPlot(estimation)
    global endTime dt
    
    figure
    
    plot(estimation.GPS(:,1),estimation.GPS(:,2),'*m', 'MarkerSize', 5);hold on;
    plot(estimation.xOdom(:,1), estimation.xOdom(:,2),'.k','MarkerSize', 10); hold on;
    plot(estimation.xEkf(:,1), estimation.xEkf(:,2),'.r','MarkerSize', 10); hold on;
    plot(estimation.xTruth(:,1), estimation.xTruth(:,2),'.b','MarkerSize', 10); hold on;
    legend('GPS Observations','Odometry Only','EKF Localization', 'Ground Truth');

    xlabel('X (meter)', 'fontsize', 12);
    ylabel('Y (meter)', 'fontsize', 12);
    grid on;
    title('Display')
    axis equal;
    print(gcf,'-dpng', '-r800', 'Display');
    
    % calculate error
    num = endTime / dt;
    step_err_Odom = zeros(num); step_err_EKF = zeros(num);
    sum_err_Odom = zeros(num); sum_err_EKF = zeros(num); 
    for i = 1: num
        step_err_Odom(i) = sqrt((estimation.xOdom(i,1)-estimation.xTruth(i,1))^2 ...
                              + (estimation.xOdom(i,2)-estimation.xTruth(i,2))^2); 
        step_err_EKF(i) = sqrt((estimation.xEkf(i,1)-estimation.xTruth(i,1))^2 ...
                             + (estimation.xEkf(i,2)-estimation.xTruth(i,2))^2);  
        if i == 1
            sum_err_Odom(i) = step_err_Odom(i);
            sum_err_EKF(i) = step_err_EKF(i);
        else          
            sum_err_Odom(i) = sum_err_Odom(i-1) + step_err_Odom(i);
            sum_err_EKF(i) = sum_err_EKF(i-1) + step_err_EKF(i);
        end
    end
    
    % 分别画出里程计和EKF的单次误差和累计误差曲线
    figure('units','normalized','position',[0.1,0.1,0.5, 0.5])
    subplot 211
    plot(step_err_Odom);
    ylabel('error');
    title('Odometry')
    legend('step error');
    grid on
    subplot 212
    plot(sum_err_Odom, 'r');
    xlabel('times');
    ylabel('error');
    legend('sum error');
    grid on
    print(gcf,'-dpng', '-r200', 'Odometry');
    
    figure('units','normalized','position',[0.1,0.1,0.5, 0.5])
    subplot 211
    plot(step_err_EKF);
    title('EKF')
    ylabel('error');
    legend('step error');
    grid on
    subplot 212
    plot(sum_err_EKF, 'r');
    xlabel('times');
    ylabel('error');
    legend('sum error');
    grid on
    print(gcf,'-dpng', '-r200', 'EKF');
    
    % 计算平均误差
    mean_err_Odom = sum_err_Odom(num) / num
    mean_err_EKF = sum_err_EKF(num) / num

end

function radian = degreeToRadian(degree)
    radian = degree/180*pi;
end