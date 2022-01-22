function [ dataNorm ] = pre_data_normalization( data, method )
% -------------------------------------------------------------------------
% Shaode Yu, nicolasyude@163.com
%   v01, 05/10/2019
%   v02, 11/06/2021
% -------------------------------------------------------------------------
%   The methods for normalization
%             (a)  'zscore':  mean=0; std=1  (default)
%             (b)   'normc':  sum( c_1 .* c_1 ) = 1
%             (c)  'linear':  (x-min (X) )/(max(X) - min(X) )
%             (d) 'clinear':  x/max(abs(X))
%   while in Mutual Information analysis, (c) is equal to (d)
% -------------------------------------------------------------------------
%   Input:
%         data, nSample * nFeature
%       method, 'zscore', 'normc', 'linear', 'clinear'
%  Output:
%     dataNorm, data after normalization
% -------------------------------------------------------------------------

% (1) check input parameters
if nargin < 2
    method = 'zscore';
end

% (2) do data normalization
[ numSample, numFeature ] = size( data );
switch lower(method)
% (2.1) z-score   
    case 'zscore'
        dataNorm = normalize( data );
        dataNorm(isnan(dataNorm)) = 0;  % Shaode Yu, 11/23/2021, zero features
% (2.2) norm colomn       
    case 'normc'
        dataNorm = normc( data );
% (2.3) linear       
    case 'linear'
        dataNorm = zeros( numSample, numFeature );
        for ii = 1:numFeature
            tmpFeature = data( :, ii );
            tmpMin = min( tmpFeature );
            tmpMax = max( tmpFeature );
            dataNorm(:,ii) = ( tmpFeature - tmpMin ) / ( tmpMax - tmpMin );
        end
% (2.4) do data normalization        
    case 'clinear' % coarse linear mapping - equal to linear in MI
        dataNorm = zeros( numSample, numFeature );
        for ii = 1:numFeature
            tmpFeature = data( :, ii );
            tmpMax = max( abs(tmpFeature) );
            dataNorm(:,ii) = ( tmpFeature ) / ( tmpMax );
        end
% (2.5) do data normalization        
    otherwise
        disp('Unknown method for data normalization and EMPTY returned. \n');
        dataNorm = [];
end
end

