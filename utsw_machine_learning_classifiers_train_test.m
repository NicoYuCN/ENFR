function [metric_test] = utsw_machine_learning_classifiers_train_test(Xtrain, Ytrain, Xtest, Ytest, ml_classifier)
% -------------------------------------------------------------------------
% Shaode Yu, yushaodemia@163.com
% -------------------------------------------------------------------------
% v01 06/24/2021
% v02 06/30/2021
% v03 11/06/2021
% v04 11/25/2021
% -------------------------------------------------------------------------
%   If there is only two parts, such as training and testing,
%       we can define the validation set the same as the testing set
% -------------------------------------------------------------------------
% to integrate classifiers (in matlab) for data prediction
% input
%   Xtrain, the data samples for model training
%           [m p] m samples and each with p features
%   Ytrain, the corresponding labels
%           [m 1] m samples and each with 1 labels in {0, 1}
%           NOTE: 0 is bad (benign) and 1 is good (malignant)
%   Xtest,  the data samples for model training
%           [k p] k samples and each with p features
%   Ytest,  the corresponding labels
%           [k 1] k samples and each with 1 labels in {0, 1}
% ml_classifier,
%          including KNN, Random Forest, Naive Bayes, Ensembles
%                    and discriminant analysis classifier
% -------------------------------------------------------------------------

% (0) to check parameter input
if nargin < 5
    ml_classifier = 'knn'; 
end

if nargin < 4
    fprintf('ERROR_ysd: no enough inputs ...\n');
    metric_test = [];    
    return;
end

% machine learning based prediction on the validation and the testing set
ml_classifier = lower(ml_classifier);
if strcmp(ml_classifier, 'knn')
    knn = ClassificationKNN.fit(Xtrain, Ytrain, 'NumNeighbors', 5); % a trick    
    predict_label_test = predict(knn, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
elseif strcmp(ml_classifier, 'rf')
    nTree = 10; % a trick
    rf = TreeBagger(nTree, Xtrain, Ytrain);    
    predict_label_test = predict(rf, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
elseif strcmp(ml_classifier, 'nb')
    % nb = NaiveBayes.fit(Xtrain, Ytrain); %fitcnb
    nb = fitcnb(Xtrain, Ytrain); %fitcnb    
    predict_label_test = predict(nb, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
elseif strcmp(ml_classifier, 'ens') % 100 learning cycles,
    ens = fitensemble(Xtrain, Ytrain, 'AdaBoostM1', 100, 'tree', 'type', 'classification'); % a trick    
    predict_label_test = predict(ens, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);

elseif strcmp(ml_classifier, 'lda')
    % 'linear' (default) | 'quadratic' | 'diaglinear' | 'diagquadratic' | 'pseudolinear' | 'pseudoquadratic'
    obj = ClassificationDiscriminant.fit(Xtrain, Ytrain);    
    predict_label_test = predict(obj, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
elseif strcmp(ml_classifier, 'lsvm')
    lsvmModel = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'linear');    
    predict_label_test = predict(lsvmModel, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
elseif strcmp(ml_classifier, 'rbfsvm')
    rbfsvmModel = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf');    
    predict_label_test = predict(rbfsvmModel, Xtest);
    metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);   

% elseif strcmp(ml_classifier, 'enfs')
%     alpha = 0.80; % If necessary
%     [B, FitInfo] = lasso(Xtrain, Ytrain, 'Alpha', alpha, 'CV', 10);    
%     idxLambda1SE = FitInfo.Index1SE;
%     coef = B(:,idxLambda1SE);
%     coef0 = FitInfo.Intercept(idxLambda1SE);
%         
%     predict_label_test = double( Xtest*coef + coef0 > 0.5);
%     metric_test = utsw_binary_classification_metrics(Ytest, predict_label_test);
    
else
    metric_test = [];
    fprintf('ERROR_ysd: an unseen classifier (knn, rf, nb, ens, lda) ...\n');
    return;
end

end

