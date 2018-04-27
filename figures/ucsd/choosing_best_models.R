library(dplyr)
library(ggplot2)

#### Random Forest ####
ucsd_auc_randfor$type <- "Random Forest"
ucsd_PR_randfor$type <- "Random Forest"
## AUC
sort(unique(ucsd_auc_randfor), decreasing = TRUE)[1:2]
# up        original
# 0.8651387 0.8548012

## PR
sort(unique(ucsd_PR_randfor), decreasing = TRUE)[1:2]
# original  weighted
# 0.5151509 0.5151509

# Original seems best.

## ROC VALUES
ucsd_randfor_original_ROC <- ucsd_randfor_results_df_roc[ucsd_randfor_results_df_roc$model == "original",]
ucsd_randfor_original_ROC$model <- "Original Random Forest"
## PR VALUES
ucsd_randfor_original_PR <- ucsd_randfor_results_df_pr[ucsd_randfor_results_df_pr$model == "original",]
ucsd_randfor_original_PR$model <- "Original Random Forest"
#### End Random Forest ####

#### GLM ####
ucsd_auc_glm$type <- "Log. Regression"
ucsd_PR_glm$type <- "Log. Regression"
## AUC
sort(unique(ucsd_auc_glm), decreasing = TRUE)[1:2]
# original  SMOTE
# 0.6902922 0.6896683
## PR
sort(unique(ucsd_PR_glm), decreasing = TRUE)[1:2]
# down       original
# 0.07834262 0.07809608
## ROC VALUES
ucsd_glm_original_ROC <- ucsd_glm_results_df_roc[ucsd_glm_results_df_roc$model == "original",]
ucsd_glm_original_ROC$model <- "Original Log. Regression"
## PR VALUES
ucsd_glm_original_PR <- ucsd_glm_results_df_pr[ucsd_glm_results_df_pr$model == "original",]
ucsd_glm_original_PR$model <- "Original Log. Regression"
#### End GLM ####

#### GBM ####
ucsd_auc_gbm$type <- "GBM"
ucsd_pr_gbm$type <- "GBM"
## AUC
sort(unique(ucsd_auc_gbm), decreasing = TRUE)[1:3]
# original        up
# 0.7765399 0.7754709
## PR
sort(unique(ucsd_pr_gbm), decreasing = TRUE)[1:3]
# original        up
# 0.2831942 0.1937535
## ROC VALUES
ucsd_gbm_original_ROC <- ucsd_gbm_results_df_roc[ucsd_gbm_results_df_roc$model == "original",]
ucsd_gbm_original_ROC$model <- "Original GBM"
## PR VALUES
ucsd_gbm_original_PR <- ucsd_gbm_results_df_pr[ucsd_gbm_results_df_pr$model == "original",]
ucsd_gbm_original_PR$model <- "Original GBM"
#### End GBM ####

#### xGBoost ####
ucsd_auc_xgboost$type <- "xGBoost"
ucsd_PR_xgboost$type <- "xGBoost"
## AUC
sort(unique(ucsd_auc_xgboost), decreasing = TRUE)[1:3]
# up        weighted  original
# 0.8249255 0.8227247 0.8218285
## PR
sort(unique(ucsd_PR_xgboost), decreasing = TRUE)[1:3]
#   weighted  original        up
# 0.4247124 0.4242364 0.3885884
## ROC VALUES
ucsd_xgboost_weighted_ROC <- ucsd_xgboost_results_df_roc[ucsd_xgboost_results_df_roc$model == "weighted",]
ucsd_xgboost_weighted_ROC$model <- "Weighted xGBoost"
## PR VALUES
ucsd_xgboost_weighted_PR <- ucsd_xgboost_results_df_pr[ucsd_xgboost_results_df_pr$model == "weighted",]
ucsd_xgboost_weighted_PR$model <- "Weighted xGBoost"
#### End xGBoost ####

#### SVM ####
ucsd_auc_svm$type <- "SVM"
ucsd_PR_svm$type <- "SVM"
## AUC
sort(unique(ucsd_auc_svm), decreasing = TRUE)[1:3]
# weighted_rad original_radial     SMOTE
#    0.7320982        0.730891 0.7131695
## PR
sort(unique(ucsd_PR_svm), decreasing = TRUE)[1:3]
# weighted_rad original_radial      SMOTE
#     0.164611        0.164502 0.07833837
## ROC VALUES
ucsd_svm_weighted_rad_ROC <- ucsd_svm_results_df_roc[ucsd_svm_results_df_roc$model == "weighted_rad",]
ucsd_svm_weighted_rad_ROC$model <- "SVM Cost-Sensitive Radial Kernel"
## PR VALUES
ucsd_svm_weighted_rad_PR <- ucsd_svm_results_df_pr[ucsd_svm_results_df_pr$model == "weighted_rad",]
ucsd_svm_weighted_rad_PR$model <- "SVM Cost-Sensitive Radial Kernel"
#### End SVM ####

#### NNET ####
ucsd_auc_nnet$type <- "ANN"
ucsd_PR_nnet$type <- "ANN"
## AUC
sort(unique(ucsd_auc_nnet), decreasing = TRUE)[1:3]
#   original  weighted        up
#  0.7188902 0.7133446 0.7083888
## PR
sort(unique(ucsd_PR_nnet), decreasing = TRUE)[1:3]
#   original         up   weighted
#  0.0985339 0.09601958 0.08602248
## ROC VALUES
ucsd_nnet_original_ROC <- ucsd_nnet_results_df_roc[ucsd_nnet_results_df_roc$model == "original",]
ucsd_nnet_original_ROC$model <- "Original ANN"
## PR VALUES
ucsd_nnet_original_PR <- ucsd_nnet_results_df_pr[ucsd_nnet_results_df_pr$model == "original",]
ucsd_nnet_original_PR$model <- "Original ANN"
#### End NNET ####


#### Combining to create Graphics #### 
ucsd_ROC_all <- rbind(ucsd_randfor_original_ROC, ucsd_glm_original_ROC,
                      ucsd_gbm_original_ROC, ucsd_xgboost_weighted_ROC,
                      ucsd_svm_weighted_rad_ROC, ucsd_nnet_original_ROC)

custom_col <- c("#000000", "red","#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_ROC_all) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(name = "ML Techniques", values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18) + 
  ggtitle("ROC Curves") + 
  theme(plot.title = element_text(lineheight=.8, size = 18),
        legend.text=element_text(size=12)) +
  labs(x = "False Positive Rate",
       y = "True Positive Rate")

ucsd_PR_all <- rbind(ucsd_randfor_original_PR, ucsd_glm_original_PR,
                     ucsd_gbm_original_PR, ucsd_xgboost_weighted_PR,
                     ucsd_svm_weighted_rad_PR, ucsd_nnet_original_PR)

ggplot(aes(x = recall, y = precision, group = model), data = ucsd_PR_all) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(name = "ML Techniques",values = custom_col) +
  ggtitle("Precision-Recall") +
  theme_bw(base_size = 18) + 
  labs(x = "Precision",
       y = "Recall") +
  theme(plot.title = element_text(lineheight=.8, size = 18),
        legend.text=element_text(size=12))
auc_table_all <- data.frame(sort(unique(ucsd_auc_randfor), decreasing = TRUE)[2],
                            sort(unique(ucsd_auc_glm), decreasing = TRUE)[1],
                            sort(unique(ucsd_auc_gbm), decreasing = TRUE)[1],
                            sort(unique(ucsd_auc_xgboost), decreasing = TRUE)[2],
                            sort(unique(ucsd_auc_svm), decreasing = TRUE)[1],
                            sort(unique(ucsd_auc_nnet), decreasing = TRUE)[1])
colnames(auc_table_all) <- c("Original Random Forest", "Original GLM",
                             "Original GBM", "Weighted xGBoost",
                             "SVM Cost-Sensitive Radial Kernel", "Original ANN")


#### Tables ####
ucsd_auc_glm <- ucsd_auc_glm %>%
  mutate(weighted = NA) %>%
  select(original, weighted, down, up, SMOTE, type)
ucsd_auc_svm <- ucsd_auc_svm %>%
  rename(weighted = weighted_lin,
         weighted_radial = weighted_rad) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
ucsd_auc_table <- rbind(ucsd_auc_randfor, ucsd_auc_xgboost, 
                        ucsd_auc_gbm, ucsd_auc_nnet,
                        ucsd_auc_glm) %>%
  mutate(weighted_radial = NA,
         original_radial = NA) %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(ucsd_auc_svm) 


ucsd_PR_glm <- ucsd_PR_glm %>%
  mutate(weighted = NA) %>%
  select(original, weighted, down, up, SMOTE, type)
ucsd_PR_svm <- ucsd_PR_svm %>%
  rename(weighted = weighted_lin,
         weighted_radial = weighted_rad) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
ucsd_PR_table <- rbind(ucsd_PR_randfor, ucsd_PR_xgboost, 
                        ucsd_pr_gbm, ucsd_PR_nnet,
                        ucsd_PR_glm) %>%
  mutate(weighted_radial = NA,
         original_radial = NA) %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(ucsd_PR_svm) 
