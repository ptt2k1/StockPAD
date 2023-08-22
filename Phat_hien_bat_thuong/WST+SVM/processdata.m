%parentDir='C:\Users\admin\3D Objects\ĐATNCN';
%list_data = ['1AAPL.csv', '2META.csv','3UBER.csv','4ADBE.csv','5MSFT.csv',...
%            '6PYPL.csv','7AMZN.csv','8BABA.csv','9DIS.csv'];

% for i = 1:length(list_data)
%     data = readtable(list_data(i));
%     
% end

data = readtable("data_svm.csv");
y = table2array(data(2:end,2:end));
t = (1:length(y(:,1)));
for i = 1:length(y(:,1))
    a(i,:) = zeros(size(y(i,:)));
    a(i,2:end) = y(i,1:end-1);
    Log_returns(i,:) = log(y(i,:)./a(i,:));
    Log_returns(i,1) = 0;
end

Data = Log_returns;
%Labels = {'N','N','UN','N','N','UN','N','N','UN'}';
%Labels = {'ARR','ARR','CHF','NSR','CHF','NSR','CHF','NSR','ARR'}';
Labels = {'UN','UN','N','UN','N','N','UN','UN','N'}';
save('SVMData5.mat',"Data","Labels");