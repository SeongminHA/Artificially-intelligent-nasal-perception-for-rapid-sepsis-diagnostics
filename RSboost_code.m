clc;
clear all;
close all;

% RSBoost algorithm (LSBoost layer + CNN-SVM layer + Hybrid Algorithms) 
% LSBoost Layer: Colorimetric sensor intensity regression tool learner

%% Regression Model Training - LSBoost Algorithm
function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
    % Ensure trainingData is provided
    if nargin < 1
        error('Training data is required as input.');
    end

    % Extract predictors and response from training data
    inputTable = trainingData;
    predictorNames = {'S.aureus-10_CFU/ml', 'S.aureus-100_CFU/ml', 'S.aureus-1000_CFU/ml', 'E.coli-10_CFU/ml', 'E.coli-100_CFU/ml', 'E.coli-1000_CFU/ml', 'P.seudomonas-10_CFU/ml', 'P.seudomonas-100_CFU/ml', 'P.seudomonas-1000_CFU/ml'};
    predictors = inputTable(:, predictorNames);
    response = inputTable.VarName12;
    isCategoricalPredictor = [false, false, false, false, false, false, false, false, false];

    % Data Preprocessing - Normalization and Outlier Removal
    predictors = normalize(predictors);
    [~, outlierIdx] = rmoutliers(response);
    predictors(outlierIdx, :) = [];
    response(outlierIdx) = [];

    % Feature Engineering - Principal Component Analysis (PCA)
    [coeff, score, ~, ~, explained] = pca(table2array(predictors));
    cumulativeVariance = cumsum(explained);
    numComponents = find(cumulativeVariance >= 98, 1); % Retain 98% variance
    predictorsPCA = score(:, 1:numComponents);

    % Template Tree Configuration
    template = templateTree('MinLeafSize', 4, 'NumVariablesToSample', 6);

    % Train LSBoost Regression Ensemble with Optimized Hyperparameters
    regressionEnsemble = fitrensemble(
        predictorsPCA, ...
        response, ...
        'Method', 'LSBoost', ...
        'NumLearningCycles', 1500, ...
        'Learners', template, ...
        'LearnRate', 0.005, ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 30));

    % Create Prediction Function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    pcaFcn = @(x) x * coeff(:, 1:numComponents);
    ensemblePredictFcn = @(x) predict(regressionEnsemble, pcaFcn(table2array(x)));
    trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

    % Store Trained Model Properties
    trainedModel.RequiredVariables = predictorNames;
    trainedModel.RegressionEnsemble = regressionEnsemble;
    trainedModel.PCACoefficients = coeff;
    trainedModel.NumComponents = numComponents;

    % Perform Cross-Validation (Monte Carlo Cross Validation)
    partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 25);

    % Calculate Validation Predictions
    validationPredictions = kfoldPredict(partitionedModel);

    % Calculate Validation RMSE
    validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'rmse'));

    % Display RMSE
    disp(['Validation RMSE: ', num2str(validationRMSE)]);
end

%% Classification Learning Using Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Hybrid Algorithms
% Load Dataset (Assuming the dataset is stored in the folder 'Nineclass')
imds = imageDatastore('Nineclass', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split Dataset into Training, Validation, and Test Sets
[imdsTrain, imdsRest] = splitEachLabel(imds, 0.6, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRest, 0.5, 'randomized');

% Display Sample Training Images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 100);
figure;
I = imtile(imds.Files(idx), 'GridSize', [10, 10]);
imshow(I);
title('Sample Training Images');

% Load Pre-trained Network (CNN-SVM Layer) - Using SqueezeNet
net = squeezenet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;

% Image Augmentation to Match Input Size of Network
imageAugmenter = imageDataAugmenter(
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandXShear', [-10, 10], ...
    'RandYShear', [-10, 10], ...
    'RandXScale', [0.7, 1.3], ...
    'RandYScale', [0.7, 1.3], ...
    'RandBrightness', [0.5, 1.5]);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% Extract Image Features Using SqueezeNet (Layer 'pool10')
layer = 'pool10';
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresValidation = activations(net, augimdsValidation, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

% Feature Scaling for SVM
scaler = fitStandardScaler(featuresTrain);
featuresTrainScaled = transform(scaler, featuresTrain);
featuresValidationScaled = transform(scaler, featuresValidation);
featuresTestScaled = transform(scaler, featuresTest);

% Train SVM Classifier on Extracted Features with Enhanced Hyperparameters
YTrain = imdsTrain.Labels;
YValidation = imdsValidation.Labels;
mdl = fitcecoc(featuresTrainScaled, YTrain, 'Learners', templateSVM('KernelFunction', 'rbf', 'BoxConstraint', 10, 'KernelScale', 'auto'));

% Hyperparameter Optimization using Bayesian Optimization
params = hyperparameters('fitcecoc', featuresTrainScaled, YTrain);
params(1).Range = [0.1, 200]; % BoxConstraint
params(2).Optimize = true; % KernelScale
mdl = fitcecoc(featuresTrainScaled, YTrain, 'Learners', templateSVM(), 'OptimizeHyperparameters', params, 'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 50));

% Validate SVM Model
YValidationPred = predict(mdl, featuresValidationScaled);
validationAccuracy = mean(YValidationPred == YValidation);
disp(['Validation Accuracy: ', num2str(validationAccuracy * 100), '%']);

% Classify Test Images Using SVM Model
YPred = predict(mdl, featuresTestScaled);
YTest = imdsTest.Labels;

% Display Test Images with Predicted Labels
idx = randi([1, max(size(imdsTest.Files))], 1, 16);
figure;
for i = 1:numel(idx)
    subplot(4, 4, i);
    I = readimage(imdsTest, idx(i));
    label = YPred(idx(i));
    imshow(I);
    title(char(label));
end

% Calculate Classification Accuracy
accuracy = mean(YPred == YTest);
disp(['Test Classification Accuracy: ', num2str(accuracy * 100), '%']);

% Ensemble Learning - Hybrid Algorithm
% Combine CNN-SVM with Random Forest and AdaBoost for Better Classification
ensembleModelRF = fitcensemble(featuresTrainScaled, YTrain, 'Method', 'Bag', 'Learners', templateTree('MinLeafSize', 5));
ensembleModelAdaBoost = fitcensemble(featuresTrainScaled, YTrain, 'Method', 'AdaBoostM1', 'Learners', templateTree('MinLeafSize', 4));

% Evaluate Ensemble Models
YEnsemblePredRF = predict(ensembleModelRF, featuresTestScaled);
ensembleAccuracyRF = mean(YEnsemblePredRF == YTest);
disp(['Random Forest Ensemble Classification Accuracy: ', num2str(ensembleAccuracyRF * 100), '%']);

YEnsemblePredAdaBoost = predict(ensembleModelAdaBoost, featuresTestScaled);
ensembleAccuracyAdaBoost = mean(YEnsemblePredAdaBoost == YTest);
disp(['AdaBoost Ensemble Classification Accuracy: ', num2str(ensembleAccuracyAdaBoost * 100), '%']);

% Stacked Ensemble Model Combining CNN-SVM, Random Forest, and AdaBoost
stackedEnsemble = fitcecoc([featuresTrainScaled, predict(ensembleModelRF, featuresTrainScaled), predict(ensembleModelAdaBoost, featuresTrainScaled)], YTrain);

% Evaluate Stacked Ensemble Model
stackedFeaturesTest = [featuresTestScaled, predict(ensembleModelRF, featuresTestScaled), predict(ensembleModelAdaBoost, featuresTestScaled)];
YStackedPred = predict(stackedEnsemble, stackedFeaturesTest);
stackedAccuracy = mean(YStackedPred == YTest);
disp(['Stacked Ensemble Classification Accuracy: ', num2str(stackedAccuracy * 100), '%']);

% Launch Regression Learner or Classification Learner App as Needed
% regressionLearner
% classificationLearner