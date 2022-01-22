
%
% Shaode Yu, yushaodemia@163.com
%
%   Purpose
%       High-dimensional small-sample-size data analysis
%           using elastic net based feature ranking
%
%   Main steps
%       (1) import dataset
%       (2) z-score data normalization
%               using
%                   pre_data_normalization.m
%       (3) elastic net based feature selection (ENFS)
%               using 
%                   func_ENFS_coef_prediction.m
%       (4) elastic net based feature ranking   (ENFR)
%               using
%                   func_ENFR_rank_importance.m
%       (5) incremental feature subsets for classification
%               using
%                   utsw_machine_learning_classifiers_train_test.m
%       (*) random data spliting
%               using
%                   utsw_random_data_spliting_train_test.m
%       (*) evaluation of binary classification performance
%               using
%                   utsw_binary_classification_metrics.m
%
% -------------------------------------------------------------------------

clear; close all; clc;

% (1) load data and get gene name (fa_gene_name, fb_patID_data, fc_patID_label)
load('data\gse10810data.mat');
% load('data\gse15852data.mat'); % another gene dataset

feature_name = fa_gene_name';
feature_beni = fb_patID_data(fc_patID_label(:,2)== 0, 2:end);
feature_mali = fb_patID_data(fc_patID_label(:,2)== 1, 2:end);
clear fa_gene_name fb_patID_data fc_patID_label

% (2) normalize the data using z-score
tmpX = pre_data_normalization([feature_beni; feature_mali], 'zscore');
feature_beni = tmpX(1:size(feature_beni,     1), :);
feature_mali = tmpX(1+size(feature_beni, 1):end, :); clear tmpX

% (3) ENFS
alpha = 0.8;
num_iteration = 50;
data_out = func_ENFS_coef_prediction(feature_beni, feature_mali, alpha, num_iteration);
coef = data_out.coef;
enfs = data_out.enfs; clear data_out

% (4) ENFR ranks
enfr_rank = func_ENFR_rank_importance(coef, num_iteration);

% (5) ENFR rank based linear SVM prediction
num_top_feature = 20;
% (5.1) ENFRq
[enfr_svm_freq] = func_ML_incremental_feature_subsets(enfr_rank.freq_rank.index, feature_beni, feature_mali, num_iteration, num_top_feature);
% (5.2) ENFRw
[enfr_svm_weit] = func_ML_incremental_feature_subsets(enfr_rank.weit_rank.index, feature_beni, feature_mali, num_iteration, num_top_feature);
% (5.3) ENFRwq
[enfr_svm_wetq] = func_ML_incremental_feature_subsets(enfr_rank.weit_freq_rank.index, feature_beni, feature_mali, num_iteration, num_top_feature);
clear feature_beni feature_mali
% (6) post-processing
clc;
% (6.1) coef analysis and ENFS
fprintf('\n------------------------------------------------------------------------------------\n');
fprintf('(1) ENFS \n');
coef_bin = double(abs(coef) > 0);
p_feature_each_iter = sum(coef_bin,2); 
p_feature_involved = sum( double(sum(coef_bin,1) > 0) );
fprintf('... max (%d), min (%d), mean (%d), std (%d), involve (%d) ...\n', ...
    max(p_feature_each_iter), min(p_feature_each_iter), ...
    round(mean(p_feature_each_iter)), round(std(p_feature_each_iter)), p_feature_involved );
fprintf('... auc (%.2f + %.2f), acc (%.2f + %.2f), sen (%.2f + %.2f), spe (%.2f + %.2f)\n', ...
    mean(enfs(:,1)), std(enfs(:,1)), mean(enfs(:,2)), std(enfs(:,2)), mean(enfs(:,3)), std(enfs(:,3)), mean(enfs(:,4) ), std(enfs(:,4)));
clear coef_bin coef p_feature_each_iter p_feature_involved
%
% (6.2) ENFR_q
fprintf('\n------------------------------------------------------------------------------------\n');
fprintf('(2) ENFR_q \n');
freqy = enfr_rank.freq_rank.freqy;
frind = enfr_rank.freq_rank.index;
for ii = 1 : num_top_feature
    fprintf('... top-(%d), \t freqy (%.2f), \t gene (%s) \n ', ii, freqy(ii), feature_name{frind(ii)} ); 
    %
    tmp_enfr_freq = squeeze(enfr_svm_freq(:,ii,:));
    fprintf('\t \t \t \t \t \t auc (%.2f + %.2f), acc (%.2f + %.2f), sen (%.2f + %.2f), spe (%.2f + %.2f)\n', ...
            mean(tmp_enfr_freq(:,1)), std(tmp_enfr_freq(:,1)), mean(tmp_enfr_freq(:,2)), std(tmp_enfr_freq(:,2)), ...
            mean(tmp_enfr_freq(:,3)), std(tmp_enfr_freq(:,3)), mean(tmp_enfr_freq(:,4) ), std(tmp_enfr_freq(:,4)));
end
clear ii freqy frind tmp*
%
% (6.3) ENFR_w
fprintf('\n------------------------------------------------------------------------------------\n');
fprintf('(3) ENFR_w \n');
weits = enfr_rank.weit_rank.weits;
wtind = enfr_rank.weit_rank.index;
for ii = 1 : num_top_feature
    fprintf('... top-(%d), \t weits (%.4f), \t gene (%s) \n ', ii, weits(ii), feature_name{wtind(ii)} ); 
    %
    tmp_enfr_weit = squeeze(enfr_svm_weit(:,ii,:));
    fprintf('\t \t \t \t \t \t auc (%.2f + %.2f), acc (%.2f + %.2f), sen (%.2f + %.2f), spe (%.2f + %.2f)\n', ...
            mean(tmp_enfr_weit(:,1)), std(tmp_enfr_weit(:,1)), mean(tmp_enfr_weit(:,2)), std(tmp_enfr_weit(:,2)), ...
            mean(tmp_enfr_weit(:,3)), std(tmp_enfr_weit(:,3)), mean(tmp_enfr_weit(:,4) ), std(tmp_enfr_weit(:,4)));
end
clear ii weits wtind tmp*
%
% (6.4) ENFR_wq
fprintf('\n------------------------------------------------------------------------------------\n');
fprintf('(4) ENFR_wq \n');
weit_freq = enfr_rank.weit_freq_rank.weit_freq;
wqind = enfr_rank.weit_freq_rank.index;
for ii = 1 : num_top_feature
    fprintf('... top-(%d), \t weit_freq (%.4f), \t gene (%s) \n ', ii, weit_freq(ii), feature_name{wqind(ii)} ); 
    %
    tmp_enfr_wetq = squeeze(enfr_svm_wetq(:,ii,:));
    fprintf('\t \t \t \t \t \t auc (%.2f + %.2f), acc (%.2f + %.2f), sen (%.2f + %.2f), spe (%.2f + %.2f)\n', ...
            mean(tmp_enfr_wetq(:,1)), std(tmp_enfr_wetq(:,1)), mean(tmp_enfr_wetq(:,2)),  std(tmp_enfr_wetq(:,2)), ...
            mean(tmp_enfr_wetq(:,3)), std(tmp_enfr_wetq(:,3)), mean(tmp_enfr_wetq(:,4) ), std(tmp_enfr_wetq(:,4)));
end
clear ii weit_freq wqind tmp*
%
