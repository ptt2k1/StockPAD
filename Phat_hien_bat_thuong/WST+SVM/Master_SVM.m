%Master SVM
%load('ECGData.mat')
ECGData = load('SVMData5.mat');

percent_train = 70;
[trainData,testData,trainLabels,testLabels] = ...
    helperRandomSplit(percent_train,ECGData);

Ctrain = countcats(categorical(trainLabels))./numel(trainLabels).*100;

Ctest = countcats(categorical(testLabels))./numel(testLabels).*100;

helperPlotRandomRecords(ECGData,14)
N = size(ECGData.Data,2);
%sn = waveletScattering('SignalLength',N, 'InvarianceScale',150,'SamplingFrequency',128);
sn = waveletScattering('SignalLength',N, 'InvarianceScale',15,'SamplingFrequency',128);

[fb,f,filterparams] = filterbank(sn);
figure
subplot(211)
plot(f,fb{2}.psift)
xlim([0 128])
grid on
title('1st Filter Bank Wavelet Filters');
subplot(212)
plot(f,fb{3}.psift)
xlim([0 128])
grid on
title('2nd Filter Bank Wavelet Filters');
xlabel('Hz');

figure(3)
phi = ifftshift(ifft(fb{1}.phift));
psiL1 = ifftshift(ifft(fb{2}.psift(:,end)));
%t = (-2^15:2^15-1).*1/128;
t = (-2^13:2^13-1).*1/128;
scalplt = plot(t,phi);
hold on
grid on
ylim([-1.5e-4 1.6e-4]);
plot([-75 -75],[-1.5e-4 1.6e-4],'k--');
plot([75 75],[-1.5e-4 1.6e-4],'k--');
xlabel('Seconds'); ylabel('Amplitude');
wavplt = plot(t,[real(psiL1) imag(psiL1)]);
legend([scalplt wavplt(1) wavplt(2)],{'Scaling Function','Wavelet-Real Part','Wavelet-Imaginary Part'});
title({'Scaling Function';'Coarsest-Scale Wavelet First Filter Bank'})
hold off

scat_features_train = featureMatrix(sn,trainData');

Nwin = size(scat_features_train,2);
scat_features_train = permute(scat_features_train,[2 3 1]);
scat_features_train = reshape(scat_features_train,...
    size(scat_features_train,1)*size(scat_features_train,2),[]);

scat_features_test = featureMatrix(sn,testData');
scat_features_test = permute(scat_features_test,[2 3 1]);
scat_features_test = reshape(scat_features_test,...
    size(scat_features_test,1)*size(scat_features_test,2),[]);

[sequence_labels_train,sequence_labels_test] = createSequenceLabels(Nwin,trainLabels,testLabels);

scat_features = [scat_features_train; scat_features_test];
allLabels_scat = [sequence_labels_train; sequence_labels_test];
rng(1);
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    scat_features, ...
    allLabels_scat, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', {'N','UN'}); %{'ARR';'CHF';'NSR'});
kfoldmodel = crossval(classificationSVM, 'KFold', 5);

predLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100;
confmatCV = confusionmat(allLabels_scat,predLabels)

fprintf('Accuracy is %2.2f percent.\n',100-loss);

%classes = categorical({'ARR','CHF','NSR'});
classes = categorical({'N','UN'});
[ClassVotes,ClassCounts] = helperMajorityVote(predLabels,[trainLabels; testLabels],classes);

MVconfmatCV = confusionmat(categorical([trainLabels; testLabels]),ClassVotes);
MVconfmatCV

model = fitcecoc(...
     scat_features_train, ...
     sequence_labels_train, ...
     'Learners', template, ...
     'Coding', 'onevsone', ...
     'ClassNames', {'N','UN'}); %{'ARR','CHF','NSR'});
predLabels = predict(model,scat_features_test);
[TestVotes,TestCounts] = helperMajorityVote(predLabels,testLabels,classes);
testaccuracy = sum(eq(TestVotes,categorical(testLabels)))/numel(testLabels)*100;
fprintf('The test accuracy is %2.2f percent. \n',testaccuracy);

confusionchart(categorical(testLabels),TestVotes)
