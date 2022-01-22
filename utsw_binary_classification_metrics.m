function metric = utsw_binary_classification_metrics(groundTruth, predictedResult)
% =========================================================================
%
% Shaode Yu (yushaodemia@163.com)
% -------------------------------------------------------------------------
% performance metrics for binary classification problems
%
% %  input parameters
%      groundTruth: the ground truth labels ([n, 1])
%      predictedResult: the predicted results ([n, 1])
%
% %  output parameters
%      metric: the 7 parameters computed for performance evaluation
% -------------------------------------------------------------------------
% v01 02/28/2019
% v02 11/16/2020
% -------------------------------------------------------------------------

% (1) to check input parameters
if nargin < 2
    disp('WRONG: No enough parameters ... \n');
    metric = [];
    return;
end

% (2) to check whether the numbers of labels match
if size(groundTruth,1) ~= size(predictedResult,1)
    metric = [];
    return;
end

% (3) performanc evaluation
testLabel = groundTruth;
preLab = predictedResult;

% (3.1) to check whether Labels are consistent
% according to random data spliting, there are two labels in certain
%     ben 0; mal 1
predictLabelx = unique(preLab);
if 1 == size(predictLabelx,1)
    if 0 == predictLabelx
        fprintf('CAUTION: benign labels only ...\n' );
    elseif 1 == predictLabelx
        fprintf('CAUTION: malignant labels only ...\n' );
    else
        fprintf('CAUTION: wrong labels %d ...\n', predictLabelx );
        metric = [];
        return;
    end
else
    if (0 ~= min(predictLabelx)) || (1 ~= max(predictLabelx))
        
        fprintf('CAUTION: benign (%d); malignant (%d) %d ...\n', ...
            min(predictLabelx), max(predictLabelx));
        
        metric = [];
        return;
    end
end

% (3.2) to baseline (ben, 0; mal, 1)
posLab = 1;
negLab = 0;

tp = sum( (preLab==posLab) & (testLabel==posLab) );
fn = sum( (preLab==negLab) & (testLabel==posLab) );
tn = sum( (preLab==negLab) & (testLabel==negLab) );
fp = sum( (preLab==posLab) & (testLabel==negLab) );

[~,~,~,auc] = perfcurve( testLabel, preLab, posLab );

acc = ( tp + tn ) / ( tp + fn + tn + fp);
sen = ( tp ) / ( tp + fn );
spe = ( tn ) / ( tn + fp );
ppv = ( tp ) / ( tp + fp );
npv = ( tn ) / ( tn + fn );
f1s = ( 2*tp ) / ( 2*tp + fp + fn );

metric = [auc acc sen spe ppv npv f1s];
end

