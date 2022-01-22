function [enfr_svm] = func_ML_incremental_feature_subsets(freq_rank_index, feature_beni, feature_mali, num_iteration, num_top_feature)
% -------------------------------------------------------------------------
% Shaode Yu, yushaodemia@163.com
%   Purpose
%       To do incremental feature subsets for machine learning based disease prediction
% -------------------------------------------------------------------------
%   Input
%       freq_rank_index, the feature rank index
%       feature_beni, benign cases [m, p], m cases with p features
%       feature_mali, malignant cases [n, p], n cases with p features
%       num_iteration, the number of iteration
%       num_top_feature, the number of top candidate features
% -------------------------------------------------------------------------
%   Output
%       metric_svm_freqy,  svm performance
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% (1) to check input parameters
if nargin < 5
    num_top_feature = 10;
end

if nargin < 4    
    fprintf('Error: No sufficient input parameters \n');
    enfr_svm = [];
    return;
end

if size(feature_beni, 2) ~= size(feature_mali, 2)
    fprintf('Error: Feature dimensions not match \n');
    enfr_svm = [];
    return;
end

% -------------------------------------------------------------------------
% (2) to do machine learning with incremental feature subsets
metric_svm_freqy = zeros( num_iteration, num_top_feature, 7); % svm performance
%
for ii = 1 : num_iteration
    % (4.1.1) random data splitting
    [train_nfog, train_fog, ttest_nfog, ttest_fog, ~, ~] = utsw_random_data_spliting_train_test(feature_beni, feature_mali);
    Xtrain = [train_nfog; train_fog];    Ytrain = [ zeros(size(train_nfog,1), 1); ones(size(train_fog,1), 1) ];
    Xttest = [ttest_nfog; ttest_fog];    Yttest = [ zeros(size(ttest_nfog,1), 1); ones(size(ttest_fog,1), 1) ];
    % (4.1.2) incremental feature selection for classification
    for jj = 1 : num_top_feature
        % re-organize the dataset
        tmp_feature_index = freq_rank_index(1:jj);
        tmp_Xtrain = Xtrain(:, tmp_feature_index);
        tmp_Xttest = Xttest(:, tmp_feature_index);
        % re-run the machine learning classifiers
        metric_svm_freqy(ii, jj, :) = utsw_machine_learning_classifiers_train_test(tmp_Xtrain, Ytrain, tmp_Xttest, Yttest, 'lsvm');
    end
    clear tmp_feature_index tmp_Xtrain tmp_Xttest
end

% -------------------------------------------------------------------------
% (3) data output
enfr_svm = metric_svm_freqy;
end

