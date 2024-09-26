clc
clear all
close all
% Modified LSBoost algorithm input
% Load workspace 
function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
[trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 16, ...
    'NumVariablesToSample', 9);
regressionEnsemble = fitrensemble(...
    predictors, ... 
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 19, ... % hyperparameter option
    'Learners', template, ...
    'LearnRate', 0.2376760912822294);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
trainedModel.RegressionEnsemble = regressionEnsemble;
% Predictor and response variable extraction
% This code processes the data into a form suitable for training a model.
inputTable = trainingData;
predictorNames = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

template = templateTree(...
    'MinLeafSize', 32, ...
    'NumVariablesToSample', 9);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 1000, ...
    'Learners', template, ...
    'LearnRate', 0.01);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
trainedModel.RegressionEnsemble = regressionEnsemble;

inputTable = trainingData;
predictorNames = {'S_aureus-10', 'S_aureus-100', 'S_aureus-1000', 'E_coli-10', 'E_coli-100', 'E_coli-1000', 'P_seudomonas-10', 'P_seudomonas-100', 'P_seudomonas-1000'};
predictors = inputTable(:, predictorNames);
response = inputTable.VarName12;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'monteCarloRun', 100);

% Calculate validation predictions
validationPredictions = monteCarloRun(partitionedModel);

% Verification RMSE Calculation
validationRMSE = sqrt(monteCarloRun(partitionedModel, 'LossFun', 'mse'));

regressionLearner

---------------------------------------------------------------------------------
%% Programmatic Transfer Learning Using Support vector machine
% LoadData
% unzip('CTCclusters.zip'); (imds(함수): 이미지 데이터의 데이터저장소- imds = imageDatastore(location, Name, Value): 하나 이상의 이름- 값 인수를 사용하여 추가 파라미터와 속성을 지정함.
imds = imageDatastore('Nineclass', ... 
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% Input layer random forest []: 행렬을 표시한다는 뜻 imds를 7(train):3(test)으로 분할하고, 3(test)는 randomized로 들어간다는 의미. 
% numel 배열요소의 개수-prod(size(A))로 쓰기도 함. 
% Idx- 최소값의 인덱싱으로 유지하기 위함. 
% I = 100개씩 랜덤하게 훈련에 들어간다는 의미.
% I의 값을 figure 로 보여라 (imshow(I)) 
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,100);

I = imtile(imds, 'Frames', idx);

figure
imshow(I)
% Load Pretrained Network
% 훈련에 사용될 사전 Network를 불러오기
net = trainNetwork(imds, layers, options);

analyzeNetwork(net) % 엔터 치고, 아래에서 스크립트 수정

inputSize = net.Layers(1).InputSize

% Extract Image Features 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool10';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
idx = [1 5 10 15 20 25 30 35 40 45 50 60 65 70 74]; %(CNN-SVM 480 x 272 이미지 추출 인덱싱)
YTest = imdsTest.Labels;
idx = [1 13 25 37 49 61 73 85 97 109 121 133 155 167 179 191]; %(CNN-SVM 1200 x 900이미지의 인덱싱)

%Fit Image Classifier
mdl = fitcecoc(featuresTrain,YTrain);

%Classify Test Images
YPred = predict(mdl,featuresTest);
idx = randi([1,max(size(imdsTest.Files))],1,16);


figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

accuracy = mean(YPred == YTest)

classificationLearner

--------------------------------------------------------------------------------------------------------
layers = [
    imageInputLayer([480 272 3])
    
    convolution2dLayer(11, 96, 'Stride', 4, 'Padding', 0)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
    
    convolution2dLayer(5, 256, 'Stride', 1, 'Padding', 2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1)
    reluLayer()
    
    convolution2dLayer(3, 384, 'Stride', 1, 'Padding', 1)
    reluLayer()
    
    convolution2dLayer(3, 256, 'Stride', 1, 'Padding', 1)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
    
    fullyConnectedLayer(4096)
    reluLayer()
    dropoutLayer(0.5)
    
    fullyConnectedLayer(4096)
    reluLayer()
    dropoutLayer(0.5)
    
    fullyConnectedLayer(numClasses)
    

%%% Step 1:  Read Images
% Read the reference image containing the object of interest.
boxImage = rgb2gray(imread('total_testset'));
figure;
imshow(boxImage);
title('image of box');

%%
% Read the target image containing a cluttered scene.
sceneImage = rgb2gray(imread('total_testset'));
figure;
imshow(sceneImage);
title('Image of a Cluttered Scene');

%% Step 2: Detect Feature Points
% Detect feature points in both images.
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

boxBriskPoints = detectBRISKFeatures(boxImage);
sceneBriskPoints = detectBRISKFeatures(sceneImage);


%%
% Visualize the Strongest feature points found in the reference image.
figure;
imshow(boxImage);
title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(boxPoints, 100));

figure;
imshow(boxImage);
title('100 Strongest Feature Points from box Image');
hold on;
plot(selectStrongest(boxBriskPoints, 100));


%%
% Visualize the strongest feature points found in the target image.
figure;
imshow(sceneImage);
title('300 Strongest Feature Points from Scene Image');
hold on;
plot(selectStrongest(scenePoints, 300));

%% Step 3: Extract Feature Descriptors
% Extract feature descriptors at the interest points in both images.
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%% Step 4: Find Putative Point Matches
% Match the features using their descriptors.
boxPairs = matchFeatures(boxFeatures, sceneFeatures);

%%
% Display putatively matched features.
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

%%
% Superpixel tool.

A = imread('total_testset');
[L,N] = superpixels(A,300);
figure
BW = boundarymask(L);
imshow(imoverlay(A,BW,'cyan'),'InitialMagnification',67)

outputImage = zeros(size(A),'like',A);
idx = label2idx(L);
numRows = size(A,1);
numCols = size(A,2);
for labelVal = 1:N
    redIdx = idx{labelVal};
    greenIdx = idx{labelVal}+numRows*numCols;
    blueIdx = idx{labelVal}+2*numRows*numCols;
    outputImage(redIdx) = mean(A(redIdx));
    outputImage(greenIdx) = mean(A(greenIdx));
    outputImage(blueIdx) = mean(A(blueIdx));
end    

figure
imshow(outputImage,'InitialMagnification',67)

A = imread('total_testset');
I = im2gray(A);
figure
imshow(I)
title('Original Image')

mask = zeros(size(I));
mask(25:end-25,25:end-25) = 1;
imshow(mask)
title('Inital Contour Location')
BW = activecontour(I, mask, 500);

imshow(BW)
title('Segmented Image, 500 Iterations')

%BW = activecontour(I,mask,500);
%imshow(BW)
%title('Segmented Image, 500 Iterations')

%%
% Jaccard index features

BW_groundTruth = imread('total_testset');

similarity = jaccard(BW, BW_groundTruth);

figure
imshowpair(BW, BW_groundTruth)
title(['Jaccard Index = ' num2str(similarity)])


%%
% feedback training CNN-SVM.

tempLayers = [
    imageInputLayer([480 272 3],"Name","data")
    convolution2dLayer([3 3],64,"Name","conv1","Stride",[2 2])
    reluLayer("Name","relu_conv1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1")
    reluLayer("Name","fire2-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1")
    reluLayer("Name","fire2-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire2-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire2-concat")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1")
    reluLayer("Name","fire3-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1")
    reluLayer("Name","fire3-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire3-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire3-concat")
    maxPooling2dLayer([3 3],"Name","pool3","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1")
    reluLayer("Name","fire4-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1")
    reluLayer("Name","fire4-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire4-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire4-concat")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1")
    reluLayer("Name","fire5-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1")
    reluLayer("Name","fire5-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire5-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire5-concat")
    maxPooling2dLayer([3 3],"Name","pool5","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1")
    reluLayer("Name","fire6-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1")
    reluLayer("Name","fire6-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire6-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire6-concat")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1")
    reluLayer("Name","fire7-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1")
    reluLayer("Name","fire7-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire7-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire7-concat")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1")
    reluLayer("Name","fire8-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1")
    reluLayer("Name","fire8-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire8-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire8-concat")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1")
    reluLayer("Name","fire9-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1")
    reluLayer("Name","fire9-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3","Padding",[1 1 1 1])
    reluLayer("Name","fire9-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire9-concat")
    dropoutLayer(0.5,"Name","drop9")
    convolution2dLayer([1 1],1000,"Name","conv10")
    reluLayer("Name","relu_conv10")
    globalAveragePooling2dLayer("Name","pool10")
    softmaxLayer("Name","prob")
    classificationLayer("Name","ClassificationLayer_predictions")];
lgraph = addLayers(lgraph,tempLayers);

svmLayer('name', 'svm') % SVM classifier layer
];
% YoloV2 layer
net = mobilenetv2();
lgraph = layerGraph(net);

imageInputSize = [1200 900 3];

imgLayer = imageInputLayer(imageInputSize,"Name","input_1")
imgLayer = 
  ImageInputLayer with properties:

                      Name: 'input_1'
                 InputSize: [1200 900 3]
        SplitComplexInputs: 0

   Hyperparameters
          DataAugmentation: 'none'
             Normalization: 'zerocenter'
    NormalizationDimension: 'auto'
                      Mean: []

lgraph = replaceLayer(lgraph,"input_1",imgLayer);

analyzeNetwork(lgraph);

featureExtractionLayer = "block_12_add";

index = find(strcmp({lgraph.Layers(1:end).Name},featureExtractionLayer));
lgraph = removeLayers(lgraph,{lgraph.Layers(index+1:end).Name});

filterSize = [7 7];
numFilters = 96;

detectionLayers = [
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv1","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch1")
    reluLayer("Name","yolov2Relu1")
    convolution2dLayer(filterSize,numFilters,"Name","yolov2Conv2","Padding", "same", "WeightsInitializer",@(sz)randn(sz)*0.01)
    batchNormalizationLayer("Name","yolov2Batch2")
    reluLayer("Name","yolov2Relu2")
    ]

numClasses = 9;

anchorBoxes = [
    16 16
    32 16
    ];

numAnchors = size(anchorBoxes,1);
numPredictionsPerAnchor = 9;
numFiltersInLastConvLayer = numAnchors*(numClasses+numPredictionsPerAnchor);

detectionLayers = [
    detectionLayers
    convolution2dLayer(1,numFiltersInLastConvLayer,"Name","yolov2ClassConv",...
    "WeightsInitializer", @(sz)randn(sz)*0.01)
    yolov2TransformLayer(numAnchors,"Name","yolov2Transform")
    yolov2OutputLayer(anchorBoxes,"Name","yolov2OutputLayer")
    ]

lgraph = addLayers(lgraph,detectionLayers);
lgraph = connectLayers(lgraph,featureExtractionLayer,"yolov2Conv1");

analyzeNetwork(lgraph)


