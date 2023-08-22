%% Input data
data = readtable('data2.csv');
data = table2array(data(:,7));
%H = table2array(data(1:2400,2));
%y = table2array(data(2401:end,2));
%load('EWA + EWC.mat');
H=1;
%y=ewa(5000:end-12);
%y1 = ewa(end-12:end);
y=data(end-48-500:end-48);
y1 = data(end-47:end);

%data = readtable('price.csv');
%y = table2array(data(1:320,2));
t = (1:length(y));
histogram(y);
ylabel('Number of Samples'), xlabel('Price');
figure

%% Sử dụng log-returns làm tham số
% a = zeros(size(y));
% a(2:end) = y(1:end-1);
% Log_returns = log(y./a);
% Log_returns(1,1) = 0;
% histogram(Log_returns);
% ylabel('Number of Samples'), xlabel('Log return')
% figure


%% XÓA CÁC GIÁ TRỊ NGOẠI LỆ 
%PLOT : using mean and standard deviation
Median = median(y);
Mad = mad(y,0);
sig = 5*Mad;
NOSD = 3; %number of standard deviations to plot
n(1:length(y)) = Median;
m(1:length(y)) = Median+NOSD*sig;
l(1:length(y)) = Median-NOSD*sig;
plot(t,y,'b ',t,n','k',t,m,'r--',t,l,'r--');
xlabel('Number of Samples'), ylabel('Price')
title('stock price with median and mad')
legend('stock price','median','limit','')
figure
%Deleting outliers
outliers = find(abs(y - Median)>NOSD*sig);
% for i = 1:length(outliers)
%     if (y(outliers(i)) > m(outliers(i)))
%         y(outliers(i)) = m(outliers(i));
%     elseif (y(outliers(i)) < l(outliers(i)))
%         y(outliers(i)) = l(outliers(i));
%     end
% end
y(outliers) = [];
%H(outliers) = [];
n1(1:length(y)) = Median;
m1(1:length(y)) = Median+NOSD*sig;
l1(1:length(y)) = Median-NOSD*sig;
t1 = (1:length(y)); 
plot(t1,y,'b ',t1,n1','k',t1,m1,'r--',t1,l1,'r--');
xlabel('Number of Samples'), ylabel('Price')
title('Log returns with median and mad after deteling outliers')
legend('log-returns','median','limit','')
figure

%% Sử dụng function_kalmanfilter
[x_hat, P, y_hat] = function_kalmanfilter(H,y);
%[x_hat, P] = logreturns_kalmanfilter(y);
%y_hat = x_hat;
%Tạo khoảng sai lệch
uncertainty_price = y_hat .* (3*sqrt(P));
curve1 = (y_hat - uncertainty_price)';
curve2 = (y_hat + uncertainty_price)';
inBetweenRegionX = [1:length(curve1), length(curve2):-1:1];
inBetweenRegionY = [curve1(1:end), fliplr(curve2(1:end))];
fill(inBetweenRegionX,inBetweenRegionY, 'y');
hold on;

plot(t1, y, 'b', t1, y_hat,'r--'); %Sử dụng t1 nếu xóa giá trị ngoại lệ
xlabel('Số lượng ngày giao dịch'), ylabel('Giá (đơn vị VND)');
title('Kết quả lọc Kalman');
legend('Vùng sai lệch','Giá thực tế', 'Giá dự đoán','Location','southeast');
xlim([1 length(t1)])
figure

%% MSE, MSA, RMSE
%MSE = sum_{i=1}^n (y - yhat)^2 /n
MSE = immse(y,y_hat);
%MAE = sum_{i=1}^n |yhat - y| /n
MAE = mae(y-y_hat);
%RMSE = sqrt(sum_{i=1}^n (y - yhat)^2 /n)
RMSE = sqrt(MSE);
fprintf('MSE = %.5f, MAE = %.5f, RMSE = %.5f\n',MSE,MAE,RMSE)

%% DỰ ĐOÁN GIÁ 12 NGÀY TIẾP THEO
last_price = y(end);
last_log_return = y;
%last_log_return(end+1) = y1(1);
predicted_prices = (1:48);
%predicted_prices(1) = y_hat(end);
%predicted_prices(1) = x_hat(end);
for i = 1:48
%     [next_log_return, nextP ] = logreturns_kalmanfilter(last_log_return);
%     next_price = last_price * exp(next_log_return(end));
    [next_log_return, nextP, next_price] = function_kalmanfilter(1,last_log_return);
    %next_price = last_price * exp(next_log_return(end));
    predicted_prices(i) = next_log_return(end);
    
    %last_log_return = next_log_return;
    last_log_return(end+1) = y1(i);
    %last_price = next_price(end);
end
plot(t1, y, 'b', t1, y_hat,'r--');
hold on
plot((length(t1)+1:length(t1)+48), y1, 'b', (length(t1)+1:length(t1)+48), predicted_prices,'m.')
xlabel('Số lượng ngày giao dịch'), ylabel('Giá (đơn vị VND)');
title('Giá dự đoán so với giá thực tế trong 48 ngày')
legend('Giá thực tế','Giá dự đoán','','Giá dự đoán 48 ngày','Location','southeast')
xlim([1 length(t1)+48])

%% MSE, MSA, RMSE
%MSE = sum_{i=1}^n (y - yhat)^2 /n
MSE_pre = immse(y1,predicted_prices');
%MAE = sum_{i=1}^n |yhat - y| /n
MAE_pre = mae(y1-predicted_prices');
%RMSE = sqrt(sum_{i=1}^n (y - yhat)^2 /n)
RMSE_pre = sqrt(MSE_pre);
fprintf('MSE_pre = %.5f, MAE_pre = %.5f, RMSE_pre = %.5f\n',MSE_pre,MAE_pre,RMSE_pre)
