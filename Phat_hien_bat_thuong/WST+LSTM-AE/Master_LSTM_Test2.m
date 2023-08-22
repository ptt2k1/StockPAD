%% Import data
% parentDir = tempdir;
parentDir='C:\Users\admin\3D Objects\ĐATNCN'; %Change this
% if exist(fullfile(parentDir,'WindTurbineHighSpeedBearingPrognosis-Data-main.zip'),'file')
%     unzip(fullfile(parentDir,'WindTurbineHighSpeedBearingPrognosis-Data-main.zip'),parentDir)
% else
%     error("File not found. "+ ...
%         "\nManually download the repository as a .zip file from GitHub. "+ ...
%         "Confirm the .zip file is in: \n%s",parentDir)
% end

% filepath = fullfile(parentDir,'WindTurbineHighSpeedBearingPrognosis-Data');
% sds = signalDatastore(filepath,'SignalVariableNames','vibration');
data = readtable("2META.csv");
% s = size(data,1);
% for i = 1:6
%     if (mod(s,6)==0)
%         break;
%     else
%        s = s - 1;
%     end
% end
data = data(:,[1,5]);
y = table2array(data(:,2));
d = flip(y);
t = (1:length(y));

%subplot(2,1,1)
figure
histogram(y);
ylabel('Number of Samples'), xlabel('Price')


%% Sử dụng log-returns làm tham số
a = zeros(size(y));
a(2:end) = y(1:end-1);
Log_returns = log(y./a);
Log_returns(1,1) = 0;



%subplot(2,1,2)
figure
histogram(Log_returns);
ylabel('Number of Samples'), xlabel('Log return')

% sds(1,:) = y(1,s/6);

% allSignals = readall(sds);
% disp(allSignals)

%% Setting
allSignals = Log_returns;

rng default
%fs = 1824; % 1AAPL
fs = 1744; %META
%fs = 1384; %uber
%fs = 1780; %msft
%fs = 1594; %pypl
%fs = 1800; %amzn
%fs = 1702; %baba
%fs= 888;
%fs = 1476; %aapl3month
%fs = 1577; %aapl
%fs = 1605;
%fs = 685;
%fs = 1300;
%fs = 1024;
%fs = 1475;
%fs = 12000;
%fs = 12000;
rdIdx = randperm(length(allSignals),1);
figure
%tiledlayout(2,3)

%% MODWPT
%for ii = 1:6
    nexttile
    [~,~,f,~,re] = modwpt(allSignals);
    %[~,~,f,~,re] = modwpt(allSignals[ii]);
    bar((f.*fs)/1e3,re.*100)
    sprintf('%2.2f',sum(re(f <= 1/4))*100)
%     record = rdIdx(ii);
%     title({'Passband Energy:' ; ['Record ' num2str(record)]}); 
    xlabel("kHz")
    ylabel("Percentage Energy")
%end

%allSignals = cellfun(@(x) resample(x,1,2),allSignals, 'UniformOutput', false);
%fs = fs/2;

%% Truc quan hoa
tstart = 0;
figure
% for ii = 1:length(allSignals)
%     t = (1:length(allSignals{ii}))./fs + tstart;    
%     hold on
%     plot(t,allSignals{ii})
%     tstart = t(end);    
% end
% hold off
t = (1:length(allSignals))./fs + tstart;    
hold on
plot(t,allSignals)
%tstart = t(end);
title("Cumulative Wind-Turbine Vibration Monitoring")
xlabel("Time (sec) -- 6 seconds per day for 50 days")
ylabel("Voltage")

%% Data Preparation and Feature Extraction
frameSize =   1*fs;
frameRate = 0.5*fs;
%nframe = (length(allSignals)-frameSize)/frameRate + 1;
%nframe = ceil((length(allSignals)-frameSize)/frameRate) + 1;
nframe = ceil((length(allSignals)-frameSize)/frameRate)+1;
nday = 1; %length(allSignals);

myXAll = zeros(frameSize,nframe*nday);
XAll = zeros(frameSize,nframe*nday);
colIdx = 1;

% 
% for ii = 1:length(allSignals)
%     XAll(:,colIdx:colIdx+nframe-1) = buffer(allSignals{ii},frameSize,...
%         frameRate,'nodelay');
%     colIdx = colIdx+nframe;
% end
XAll(:,colIdx:colIdx+nframe-1) = buffer(allSignals,frameSize,...
        floor(frameRate),'nodelay');

XAll = single(XAll);

%% Wavelet
N = size(XAll,1);
sn = waveletScattering('SignalLength',N,'SamplingFrequency', fs,...
    'InvarianceScale',0.2,'QualityFactors', [4 1],...
   'OversamplingFactor', 1,'Precision', 'single');

%NOT SO IMPORTANT, ignore!
% [~,pathbyLev] = paths(sn);
Ncfs = numCoefficients(sn)
% sum(pathbyLev)

%GPU run with CUDA, not needed.
% useGPU = true;
% if useGPU
%     XAll = gpuArray(XAll);
% end

SAll = sn.featureMatrix(XAll);
SAll = SAll(2:end,:,:);
npaths = size(SAll,1);
scatfeatures = squeeze(num2cell(SAll,[1,2]));

%% Chia train, valid, test
%ntrain = 3/11;
%ntrain = 7/12;
%ntrain = 1/9;
ntrain = 5;
nvalid = 5;
trainscatFeatures = scatfeatures(1:ntrain);
validationscatFeatures = scatfeatures(ntrain+1:ntrain+nvalid);
testscatFeatures = scatfeatures((ntrain+nvalid+1):end);
% validationscatFeatures = scatfeatures(ntrain*nframe+1:ntrain*nframe+1/18*nframe);
% testscatFeatures = scatfeatures((ntrain*nframe+1/18*nframe+1):end);


% rawfeatures = num2cell(XAll,1)';
% rawfeatures = cellfun(@transpose,rawfeatures, 'UniformOutput', false);
% rawABSfeatures = cellfun(@abs,rawfeatures, 'UniformOutput', false);
%ntrain = 3/11;
% trainrFeatures = rawABSfeatures(1:ntrain*nframe);
% validationrFeatures = rawABSfeatures(ntrain*nframe+1:ntrain*nframe+3/11*nframe);
% testrFeatures = rawABSfeatures((ntrain*nframe+3/11*nframe+1):end);
% validationrFeatures = rawABSfeatures(ntrain*nframe+1:ntrain*nframe+1/18*nframe);
% testrFeatures = rawABSfeatures((ntrain*nframe+1/18*nframe+1):end);

%% Deep Networks
Ntimesteps = Ncfs;
lstmAutoEncoder = [ sequenceInputLayer(npaths, 'Normalization','rescale-symmetric',... %change from zscore to rescale-symmetric
                                       'Name','input','Min',Ntimesteps)
    lstmLayer(npaths,'Name','lstm1a')
    reluLayer('Name','relu1')
    lstmLayer(floor(npaths/2),'Name','lstm2a','OutputMode','sequence')%changed from last to sequence
    dropoutLayer(0.2,'Name','drop1')
    reluLayer('Name','relu2')
%     repeatVectorLayer(Ntimesteps)
    lstmLayer(floor(npaths/2),'Name','lstm2b')
    dropoutLayer(0.2,'Name','drop2')
    reluLayer('Name','relu3')
    lstmLayer(npaths,'Name','lstm1b')
    reluLayer('Name','relu4')
    regressionLayer('Name','regression') ];
%Train
options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch',...
    'ValidationData',{validationscatFeatures,validationscatFeatures},...
    'ValidationFrequency',20,...
    'Plots','training-progress',...
    'Verbose', false);
%     'OutputNetwork','best-validation-loss');

%% Trainning
scatLSTMAutoencoder = trainNetwork(trainscatFeatures,trainscatFeatures,...
    lstmAutoEncoder,options);

%% Threshold determination
ypredTrain = cellfun(@(x) predict(scatLSTMAutoencoder,x),trainscatFeatures,'UniformOutput',false);
maeTrain = cellfun(@(x,y) maeLoss(x,y),ypredTrain,trainscatFeatures);
ypredValidation = cellfun(@(x) predict(scatLSTMAutoencoder,x),validationscatFeatures,'UniformOutput',false);
maeValid = cellfun(@(x,y) maeLoss(x,y),ypredValidation,validationscatFeatures);
ypredTest = cellfun(@(x) predict(scatLSTMAutoencoder,x),testscatFeatures,'UniformOutput',false);
maeTest = cellfun(@(x,y) maeLoss(x,y),ypredTest,testscatFeatures);
% if useGPU
%     [maeTrain,maeValid,maeTest] = gather(maeTrain,maeValid,maeTest);
% end

thresh = quantile(maeValid,0.75)+1.5*iqr(maeValid);

figure
% plot(...
%     (1:length(maeTrain))/11,maeTrain,'b',...
%     (length(maeTrain)+[1:length(maeValid)])/11,maeValid,'g',...
%     (length(maeTrain)+length(maeValid)+[1:length(maeTest)])/11,maeTest,'r',...
%     'linewidth',1.5)
% hold on
% plot((1:550)/11,thresh*ones(550,1),'k')

% plot(...
%     (1:length(maeTrain))/nframe,maeTrain,'b',...
%     (length(maeTrain)+[1:length(maeValid)])/nframe,maeValid,'g',...
%     (length(maeTrain)+length(maeValid)+[1:length(maeTest)])/nframe,maeTest,'r',...
%     'linewidth',1.5)
% plot(...
%     (1:length(maeTrain)),maeTrain,'b',...
%     (length(maeTrain)+[1:length(maeValid)]),maeValid,'g',...
%     (length(maeTrain)+length(maeValid)+[1:length(maeTest)]),maeTest,'r',...
%     'linewidth',1.5)
mae_all = vertcat(maeTrain, maeValid, maeTest);
z = zeros(size(mae_all));
z(mae_all > thresh) = max(mae_all) + 0.000005;
z(mae_all < thresh) = min(mae_all) - 0.000005;

z1 = zeros(size(mae_all));
z1(mae_all > thresh) = max(y) + 1;
z1(mae_all < thresh) = min(y) - 1;

t_z1 = (1: length(z1));
z1 = z1';

idx=0;
for i = 1: length(z1)-1
    if (z1(i) ~= z1(i+1))
        idx = idx+1;
        index(idx) = i;
    end
end

%if (length(index) >= 1)
    t_z1(end+1:end+length(index)) = index;

t_z1 = sort(t_z1);

for i = 1:length(index)
    z1 = [z1(1:index(i)) z1(index(i)+1) z1(index(i)+1:end)];
    if (i < length(index))
        index(i+1) = index(i+1) + i;
    end
end
%end
z = single(z);

maeTrain(end+1) = maeValid(1);
maeValid(end+1) = maeTest(1);

plot(...
    (1:length(maeTrain)),maeTrain,'b',...
    (length(maeTrain)-1+[1:length(maeValid)]),maeValid,'g',...
    (length(maeTrain)+length(maeValid)-2+[1:length(maeTest)]),maeTest,'r',...
    'linewidth',1)
hold on    

%plot((1:nframe*nday)/nframe,thresh*ones(nframe*nday,1),'k')
plot((1:nframe*nday),thresh*ones(nframe*nday,1),'k')
%plot((1:nframe*63)/nframe,thresh*ones(nframe*63,1),'k')
hold on
% mae_all = vertcat(maeTrain, maeValid, maeTest);
% z = zeros(size(mae_all));
% z(mae_all > thresh) = max(mae_all) + 0.000005;
% z(mae_all < thresh) = min(mae_all) - 0.000005;
% 
% z = single(z);
%plot((1:length(z)),z,'m','linewidth',1);

hold off
xlabel("Time (day)")
ylabel("MAE")
xlim([1 nframe])
legend("Training","Validation","Test","","Thresh","Location","NorthWest")
title("LSTM Autoencoder with Wavelet Scattering Sequences")
grid on

%Continue...
%The following LSTM autoencoder was trained on the raw data.
k = length(y)/nframe;
ntrain = ntrain +1;
figure
plot(...
    ((1:length(d(1:k*ntrain)))/k),d(1:k*ntrain),'b',...
    ((length(d(1:k*ntrain))-1+[1:length(d(k*ntrain+1:k*(ntrain+nvalid)))])/k),d(k*ntrain+1:k*(ntrain+nvalid)),'g',...
    ((length(d(1:k*ntrain))+length(d(k*ntrain+1:k*(ntrain+nvalid)))-2+[1:length(d(k*(ntrain+nvalid)+1:end))])/k),d(k*(ntrain+nvalid)+1:end),'r',...
    'linewidth',1)
hold on
% mae_all = vertcat(maeTrain, maeValid, maeTest);
% z = zeros(size(mae_all));
% z(mae_all > thresh) = max(mae_all) + 0.000005;
% z(mae_all < thresh) = min(mae_all) - 0.000005;
% 
% z = single(z);
%plot((1:length(z1)),z1,'m','linewidth',1);
plot(t_z1,z1,'m','linewidth',1);

hold off
xlabel("Time (day)")
ylabel("Price (USD)")
xlim([1 nframe])
legend("Training","Validation","Test","Window","Location","NorthWest")
title("LSTM Autoencoder with Price Stock")
grid on