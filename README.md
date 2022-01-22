# ENFR: elastic net based feature ranking

The code is for "High-dimensional small-sample-size data analysis using elastic net based feature ranking"

 Shaode Yu, yushaodemia@163.com

   Purpose
       High-dimensional small-sample-size data analysis
           using elastic net based feature ranking

   Main steps
       (1) import dataset
       (2) z-score data normalization
               using
                   pre_data_normalization.m
       (3) elastic net based feature selection (ENFS)
               using 
                   func_ENFS_coef_prediction.m
       (4) elastic net based feature ranking   (ENFR)
               using
                   func_ENFR_rank_importance.m
       (5) incremental feature subsets for classification
               using
                   utsw_machine_learning_classifiers_train_test.m
       (*) random data spliting
               using
                   utsw_random_data_spliting_train_test.m
       (*) evaluation of binary classification performance
               using
                   utsw_binary_classification_metrics.m

