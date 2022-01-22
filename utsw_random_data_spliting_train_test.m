function [train_ben, train_mal, ttest_ben, ttest_mal, ...
          ttest_ben_index, ttest_mal_index] = ...
          utsw_random_data_spliting_train_test(data_ben, data_mal, case_ratio, flag_ratio)
% =========================================================================
%
% Shaode Yu (yushaodemia@163.com)
% -------------------------------------------------------------------------
% random data splitting
%
% %  input parameters
%      data_ben: benign samples ([m, p]) m samples and each with p features
%      data_mal: malignant samples ([n, p]) n samples and each with p features
%      case_ratio: the ratio of each groups for training (default 80% for training)
%      flag_ratio: 1 (the same number of cases in each group for training, default)
%                  0 (the sampe ratio of each group for training)
%
% %  output parameters
%      train_ben: the benign samples in the training set
%      train_mal: the malignant samples in the training set
%      ttest_ben: the benign samples in the testing set
%      ttest_mal: the malignant samples in the testing set
%      ttest_ben_index: the index of benign samples in the testing set
%      ttest_mal_index: the index of malignant samples in the testing set
% -------------------------------------------------------------------------
% v01 10/08/2020
% v02 11/16/2020
% v03 04/10/2021
% -------------------------------------------------------------------------

% (1) to check input parameters
if nargin < 4
    flag_ratio = 1; 
    % 1, the same number of cases in each group for training
    % 0, the ratio of each group
end
if nargin < 3
    case_ratio = 0.8;
    % 80% for training
end

if nargin < 2
    fprintf('ERROR: insufficient input parameters ...\n');
    train_ben = []; train_mal = [];
    ttest_ben = []; ttest_mal = [];
    ttest_ben_index = [];
    ttest_mal_index = [];
    return;
end

% (2) to check whether feature dimensions match
if size(data_ben,2) ~= size(data_mal,2)
    fprintf('ERROR: feature dimension not match ...\n');
    train_ben = []; train_mal = [];
    ttest_ben = []; ttest_mal = [];
    ttest_ben_index = [];
    ttest_mal_index = [];
    return;
end

% (3) random data splitting
num_ben = size(data_ben,1);
num_mal = size(data_mal,1);
% (3.1) to determine the number of cases for training
if 1 == flag_ratio
    case_base_ben = round( case_ratio * min([num_ben, num_mal]) );
    % to select the group with fewer cases as the baseline
    case_base_mal = case_base_ben;
    % to keep the number of cases from each group the same
else
    case_base_ben = round( case_ratio * num_ben );
    case_base_mal = round( case_ratio * num_mal );
    % to keep the ratio of cases from each group the same
end

% (3.2) to split the cases into training and testing sets
rand_ben = randperm(num_ben);
rand_mal = randperm(num_mal);

% (3.2.1) the training set
train_ben = data_ben(rand_ben(1:case_base_ben), :);
train_mal = data_mal(rand_mal(1:case_base_mal), :);

% (3.2.2) the testing set
ttest_ben = data_ben(rand_ben(1+case_base_ben:end), :);
ttest_mal = data_mal(rand_mal(1+case_base_mal:end), :);

% (3.2.3) the index for the testing set
%             if the wrong classification cases are interested for analysis
ttest_ben_index = rand_ben(1+case_base_ben:end);
ttest_mal_index = rand_mal(1+case_base_mal:end);
end

