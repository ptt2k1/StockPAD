function [x_hat, P, y_hat] = function_kalmanfilter(H, y)
    n = size(y);
    x_pre = NaN(n); x_pre(1) = y(1)/H(1);
    P_pre = NaN(n); P_pre(1) = 0;
    
    x_hat = NaN(n); %x_hat(1) = y(1); y_hat = H*
    P = NaN(n);%P(1) = 0;

    y_hat=NaN(n); 
    I = NaN(n);
    S = NaN(n);

    Q = 1e-4;
    R = 0.001;

    for k = 1:length(y)
        %Predict
        if (k>1)
            x_pre(k) = x_hat(k - 1);
            P_pre(k) = P(k - 1) + Q;
        end
    
        %Update
        y_hat(k) = H*x_pre(k);
        I(k) = y(k) - y_hat(k); %Phần dư của phép đo
        S(k) = H*P_pre(k)*H' + R; %Hiệp phương sai phần dư
        K = P_pre(k)*H'/S(k); %Kalman gain

        x_hat(k) = x_pre(k) + K*I(k);
        P(k) = P_pre(k) - K*H*P_pre(k);
    end