library(dplyr)
library(ggplot2)
library(gridExtra)


plot_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

#### Random Forest ####
cr_card_auc_randfor$type <- "Random Forest"
cr_card_PR_randfor$type <- "Random Forest"
## AUC
sort(unique(cr_card_auc_randfor), decreasing = TRUE)[1:4]
#       down     SMOTE  original
# 0.9682755 0.9638472 0.9543801
## PR
sort(unique(cr_card_PR_randfor), decreasing = TRUE)[1:5]
#        up  original  weighted     SMOTE
# 0.8527693 0.8330339 0.8330339 0.8110314

# Original seems best.

## ROC VALUES
cr_card_randfor_original_ROC <- cr_card_rf_results_df_roc[cr_card_rf_results_df_roc$model == "original",]
cr_card_randfor_original_ROC$model <- "Original Random Forest"
## PR VALUES
cr_card_randfor_original_PR <- cr_card_rf_results_df_pr[cr_card_rf_results_df_pr$model == "original",]
cr_card_randfor_original_PR$model <- "Original Random Forest"
#### End Random Forest ####

#### GLM ####
cr_card_auc_glm$type <- "Log. Regression"
cr_card_PR_glm$type <- "Log. Regression"
## AUC
sort(unique(cr_card_auc_glm), decreasing = TRUE)[1:4]
# original     SMOTE        up
# 0.9443036 0.9088207 0.8967922
## PR
sort(unique(cr_card_PR_glm), decreasing = TRUE)[1:4]
# original        up      SMOTE
# 0.758459 0.7249762 0.03751652
## ROC VALUES
cr_card_glm_original_ROC <- cr_card_glm_results_df_roc[cr_card_glm_results_df_roc$model == "original",]
cr_card_glm_original_ROC$model <- "Original Log. Regression"
## PR VALUES
cr_card_glm_original_PR <- cr_card_glm_results_df_pr[cr_card_glm_results_df_pr$model == "original",]
cr_card_glm_original_PR$model <- "Original Log. Regression"
#### End GLM ####

#### GBM ####
cr_card_auc_gbm$type <- "GBM"
cr_card_PR_gbm$type <- "GBM"
## AUC
sort(unique(cr_card_auc_gbm), decreasing = TRUE)[1:4]
# weighted      down       up
# 0.9744766 0.9703028 0.969084
## PR
sort(unique(cr_card_PR_gbm), decreasing = TRUE)[1:4]
#       up  weighted     SMOTE
# 0.8358836 0.8202291 0.5228063
## ROC VALUES
cr_card_gbm_original_ROC <- cr_card_gbm_results_df_roc[cr_card_gbm_results_df_roc$model == "weighted",]
cr_card_gbm_original_ROC$model <- "Weighted GBM"
## PR VALUES
cr_card_gbm_original_PR <- cr_card_gbm_results_df_pr[cr_card_gbm_results_df_pr$model == "weighted",]
cr_card_gbm_original_PR$model <- "Weighted GBM"
#### End GBM ####

#### xGBoost ####
cr_card_auc_xgboost$type <- "xGBoost"
cr_card_PR_xgboost$type <- "xGBoost"
## AUC
sort(unique(cr_card_auc_xgboost), decreasing = TRUE)[1:4]
#        up  original     SMOTE
# 0.9771028 0.9692195 0.9673158
## PR
sort(unique(cr_card_PR_xgboost), decreasing = TRUE)[1:4]
#  original weighted     SMOTE
# 0.8336317 0.806801 0.7895484
## ROC VALUES
cr_card_xgboost_weighted_ROC <- cr_card_xgboost_results_df_roc[cr_card_xgboost_results_df_roc$model == "original",]
cr_card_xgboost_weighted_ROC$model <- "Original xGBoost"
## PR VALUES
cr_card_xgboost_weighted_PR <- cr_card_xgboost_results_df_pr[cr_card_xgboost_results_df_pr$model == "original",]
cr_card_xgboost_weighted_PR$model <- "Original xGBoost"
#### End xGBoost ####

#### SVM ####
cr_card_auc_svm$type <- "SVM"
cr_card_PR_svm$type <- "SVM"
## AUC
sort(unique(cr_card_auc_svm), decreasing = TRUE)[1:8]
# type weighted_rad original_rad  weighted  original      down        up     SMOTE
#  SVM    0.9627719    0.9592933 0.9247987 0.9181649 0.9181649 0.9181649 0.9181649
## PR
sort(unique(cr_card_PR_svm), decreasing = TRUE)[1:8]
# type  original      down        up     SMOTE  weighted weighted_rad original_rad
#  SVM 0.7810338 0.7810338 0.7810338 0.7810338 0.7719868    0.7437641    0.7322844

# Weighted Radial Kernel performs superiorly in terms of ROC AUC, but falls short
# when taking the Precision-Recall curves into consideration. The linear models, especially
# the cost-sensitive one performs well in terms of ROC AUC, but does not show
# strong performance on the PR scale.

## ROC VALUES
cr_card_svm_weighted_rad_ROC <- cr_card_svm_results_df_roc[cr_card_svm_results_df_roc$model == "weighted",]
cr_card_svm_weighted_rad_ROC$model <- "SVM Cost-Sensitive Linear"
## PR VALUES
cr_card_svm_weighted_rad_PR <- cr_card_svm_results_df_pr[cr_card_svm_results_df_pr$model == "weighted",]
cr_card_svm_weighted_rad_PR$model <- "SVM Cost-Sensitive Linear"
#### End SVM ####

#### NNET ####
cr_card_auc_nnet$type <- "ANN"
cr_card_PR_nnet$type <- "ANN"
## AUC
sort(unique(cr_card_auc_nnet), decreasing = TRUE)[1:4]
# type  weighted     down  original
# ANN 0.9737695 0.970828 0.9639963
## PR
sort(unique(cr_card_PR_nnet), decreasing = TRUE)[1:4]
# type  original  weighted     SMOTE
# ANN 0.7773746 0.7618862 0.6677604
## ROC VALUES
cr_card_nnet_original_ROC <- cr_card_nnet_results_df_roc[cr_card_nnet_results_df_roc$model == "weighted",]
cr_card_nnet_original_ROC$model <- "Weighted ANN"
## PR VALUES
cr_card_nnet_original_PR <- cr_card_nnet_results_df_pr[cr_card_nnet_results_df_pr$model == "weighted",]
cr_card_nnet_original_PR$model <- "Weighted ANN"
#### End NNET ####


#### Combining to create Graphics #### 
cr_card_ROC_all <- rbind(cr_card_randfor_original_ROC, cr_card_glm_original_ROC,
                         cr_card_gbm_original_ROC, cr_card_xgboost_weighted_ROC,
                         cr_card_svm_weighted_rad_ROC, cr_card_nnet_original_ROC)

custom_col <- c("#000000", "red","#009E73", "#0072B2", "#D55e00", "#CC79A7")

roc_curve <- ggplot(aes(x = fpr, y = tpr, group = model), data = cr_card_ROC_all) +
  geom_line(aes(color = model), size = 1.2) +
  scale_color_manual(name = "",values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1.2) +
  theme_bw(base_size = 18) + 
  ggtitle("ROC Curves") + 
  theme(plot.title = element_text(lineheight = 1,size = 26),
        legend.position="bottom",
        legend.direction="vertical",
        legend.text=element_text(size=30), 
        legend.key = element_rect(size = 10),
        legend.key.size = unit(2.5, 'lines')) +
  guides(colour = guide_legend(override.aes = list(size=3),ncol=3)) +
  labs(x = "False Positive Rate",
       y = "True Positive Rate")

cr_card_PR_all <- rbind(cr_card_randfor_original_PR, cr_card_glm_original_PR,
                     cr_card_gbm_original_PR, cr_card_xgboost_weighted_PR,
                     cr_card_svm_weighted_rad_PR, cr_card_nnet_original_PR)

pr_curve <- ggplot(aes(x = recall, y = precision, group = model), data = cr_card_PR_all) +
  geom_line(aes(color = model), size = 1.2) +
  scale_color_manual(name = "",values = custom_col) +
  ggtitle("Precision-Recall") +
  theme_bw(base_size = 18) + 
  labs(x = "Precision",
       y = "Recall") +
  guides(colour = guide_legend(override.aes = list(size=3)),
         fill=guide_legend(ncol=3)) +
  theme(plot.title = element_text(lineheight = 1,size = 26),
        legend.direction="vertical")

my_legend <- plot_legend(roc_curve)

pr_roc_graph <- grid.arrange(arrangeGrob(roc_curve + theme(legend.position = "none"),
                                         pr_curve + theme(legend.position = "none"),
                                         nrow = 1),
                             my_legend, nrow = 2, heights = c(20,5))
####
auc_table_all <- data.frame(sort(unique(cr_card_auc_randfor), decreasing = TRUE)[4],
                            sort(unique(cr_card_auc_glm), decreasing = TRUE)[2],
                            sort(unique(cr_card_auc_gbm), decreasing = TRUE)[2],
                            sort(unique(cr_card_auc_xgboost), decreasing = TRUE)[3],
                            sort(unique(cr_card_auc_svm), decreasing = TRUE)[4],
                            sort(unique(cr_card_auc_nnet), decreasing = TRUE)[2])
colnames(auc_table_all) <- c("Original Random Forest", "Original GLM",
                             "Original GBM", "Weighted xGBoost",
                             "SVM Cost-Sensitive Radial Kernel", "Original ANN")


#### Tables ####
cr_card_auc_glm <- cr_card_auc_glm %>%
  mutate(weighted = " ") %>%
  select(original, weighted, down, up, SMOTE, type)
cr_card_auc_svm <- cr_card_auc_svm %>%
  dplyr::rename(weighted_radial = weighted_rad,
                original_radial = original_rad) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
cr_card_auc_table <- rbind(cr_card_auc_randfor, cr_card_auc_xgboost, 
                           cr_card_auc_gbm, cr_card_auc_nnet,
                           cr_card_auc_glm) %>%
  mutate(weighted_radial = " ",
         original_radial = " ") %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(cr_card_auc_svm) 


cr_card_PR_glm <- cr_card_PR_glm %>%
  mutate(weighted = " ") %>%
  select(original, weighted, down, up, SMOTE, type)
cr_card_PR_svm <- cr_card_PR_svm %>%
  dplyr::rename(weighted_radial = weighted_rad,
         original_radial = original_rad) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
cr_card_PR_table <- rbind(cr_card_PR_randfor, cr_card_PR_xgboost, 
                       cr_card_PR_gbm, cr_card_PR_nnet,
                       cr_card_PR_glm) %>%
  mutate(weighted_radial = " ",
         original_radial = " ") %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(cr_card_PR_svm) 
