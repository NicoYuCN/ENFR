function [enfr] = func_ENFR_rank_importance(coef_matrix, num_iteration)
% =========================================================================
%
% Shaode Yu (yushaodemia AT 163 DOT com)
% -------------------------------------------------------------------------
% feature ranking based on the frequency, weights, 
%       and weighted frequency of selected features via elastic net
%
% %  input parameters
%      coef_matrix: estimated coefficients matrix from 
%                     multiple times of elastic net based feature selection
%      num_iteration: number of iteration of ENFS
%
% %  output parameters
%       enfr
%           enfr.index_freq_rank: frequency based feature ranks
%           enfr.index_weit_rank: weights based feature ranks
%           enfr.index_weit_freq_rank: weighted frequency based feature ranks
% -------------------------------------------------------------------------
% v01 11/16/2020
% v02 09/03/2021
% v03 11/27/2021
% v04 01/22/2022
% -------------------------------------------------------------------------

% (1) to check the parameters

% (2) to extract the feature weights
coef = coef_matrix;

% (3) to count the frequency of selected features
freCoef = sum( abs(coef)>0 ,1)/num_iteration; % coefficients not equal to 0
% (3.1) frequency based feature ranking 
[fea_freqy, fea_freqy_index] = sort(freCoef,'descend');
% here we get the frequency, and corresponding index
% (3.2) output of frequency based feature ranking
index_freq_rank.freqy = fea_freqy; 
index_freq_rank.index = fea_freqy_index;

% (4) absolute weights based feature ranking
coef_sum = sum(coef, 1); % sum of the coefficient weights
coef_sum = abs(coef_sum)/num_iteration; 
% (4.1) absolute weights based feature ranking 
[fea_weit, fea_weit_index] = sort(coef_sum,'descend');
% here we get the absolute weights, and corresponding index
% (4.2) output of absolute weights based feature ranking
index_weit_rank.weits = fea_weit; 
index_weit_rank.index = fea_weit_index;

% (5) ahsolute weightes weighted frequency based feature ranking
weit_freq = freCoef .* coef_sum;
% (5.1) absolute weights weighted frequency based feature ranking 
[fea_weit_freq, fea_weit_freq_index] = sort(weit_freq,'descend');
% here we get the absolute weights, and corresponding index
% (5.2) output of absolute weights based feature ranking
index_weit_freq_rank.weit_freq = fea_weit_freq; 
index_weit_freq_rank.index = fea_weit_freq_index;

% (6) save data and out
enfr.freq_rank = index_freq_rank;
enfr.weit_rank = index_weit_rank;
enfr.weit_freq_rank = index_weit_freq_rank;


end

