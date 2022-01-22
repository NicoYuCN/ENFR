function [data_out] = func_ENFS_coef_prediction(in_data_ben, in_data_mal, alpha, num_iteration)
% -------------------------------------------------------------------------
% Shaode Yu (yushaodemia AT 163 DOT com)
%   Purpose
%       elastic net based feature selection and weighting
% -------------------------------------------------------------------------
%  Input parameters
%      in_data_ben: the input beni cases
%      in_data_mal: the input mali cases
%         Note, benign 0; malignant 1
%     alpha: the value to balance L1 and L2 penalty
%     num_iteration: the number of iteration
% -------------------------------------------------------------------------
% Output parameters
%   data_out
%      data_out.coef, the coefficient matrix
%      data_out.enfs, ENFS based prediction
%                       
% -------------------------------------------------------------------------
% v01 05/28/2020
% v02 11/16/2020
% v03 11/26/2021
% V04 01/22/2022
% -------------------------------------------------------------------------

% (1) to check input parameters
% (1.1) to check input parameters
if nargin < 4
    num_iteration = 30;
end

if nargin < 3
    alpha = 0.8;
end

if nargin < 2
    fprintf('ERROR: insufficient input parameters ...\n');
    data_out.coef = [];
    data_out.enfs = [];
    return;
end

% (1.2) to check whether input data is correct
if size(in_data_ben, 2) ~= size(in_data_mal, 2)
    fprintf('ERROR: feature dimension not match ...\n');
    data_out.coef_matrix = [];
    data_out.enfs = [];
    return;
end
% -------------------------------------------------------------------------

% (2) to start offline elastic net based feature ranking and prediction
fprintf('... start elastic net based feature selection \n');
% (2.1) to define the output matrix
metric = zeros(num_iteration, 7);
coeffx = zeros(num_iteration, size(in_data_ben,2));
%
% (2.2) to retrieve the source data
data_ben = in_data_ben;  
data_mal = in_data_mal;
% (2.3) to start the iteration
for ii = 1 : num_iteration    
    % (2.3.1) random data splitting
    [train_ben, train_mal, ttest_ben, ttest_mal, ~, ~] = utsw_random_data_spliting_train_test(data_ben, data_mal); % 80% for training and rest for testing
    % (2.3.2) data re-organization
    XTrain = [train_ben; train_mal];    yTrain = [zeros(size(train_ben,1), 1); ones(size(train_mal,1), 1)];
    XTest = [ttest_ben; ttest_mal];     yTest  = [zeros(size(ttest_ben,1), 1); ones(size(ttest_mal,1), 1)];
    % (2.3.3) 10-folder cross-validation of elastic net
    [B,FitInfo] = lasso(XTrain,yTrain,'Alpha',alpha,'CV',10);    
    idxLambda1SE = FitInfo.Index1SE;
    coef = B(:,idxLambda1SE);
    coef0 = FitInfo.Intercept(idxLambda1SE);
    % (2.3.4) on the testing
    yhat = XTest*coef + coef0;
    yhat = double(yhat>0.5);
    metric(ii,:) = utsw_binary_classification_metrics(yTest, yhat);
    % (2.3.5) on the coefficients
    coeffx(ii,:) = coef';
    fprintf('...... (%d)/(%d) \n', ii, num_iteration);
end
% -------------------------------------------------------------------------

% (3) to save data
data_out.coef = coeffx;
data_out.enfs = metric;
end

