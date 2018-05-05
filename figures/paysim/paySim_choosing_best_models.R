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
paySim_auc_randfor$type <- "Random Forest"
paySim_PR_randfor$type <- "Random Forest"
## AUC
sort(unique(paySim_auc_randfor), decreasing = TRUE)[1:4]
#          type  original  weighted        up
# Random Forest 0.9999954 0.9999954 0.9999942
## PR
sort(unique(paySim_PR_randfor), decreasing = TRUE)[1:5]
#          type  original  weighted        up     SMOTE
# Random Forest 0.9983619 0.9983619 0.9980407 0.8255387

# Original seems best.

## ROC VALUES
paySim_randfor_original_ROC <- paySim_rf_results_df_roc[paySim_rf_results_df_roc$model == "original",]
paySim_randfor_original_ROC$model <- "Original Random Forest"
## PR VALUES
paySim_randfor_original_PR <- paySim_rf_results_df_pr[paySim_rf_results_df_pr$model == "original",]
paySim_randfor_original_PR$model <- "Original Random Forest"
#### End Random Forest ####

#### GLM ####
paysim_auc_glm$type <- "Log. Regression"
paySim_PR_glm$type <- "Log. Regression"
## AUC
sort(unique(paysim_auc_glm), decreasing = TRUE)[1:4]
#            type        up  original     SMOTE
# Log. Regression 0.9939988 0.8717525 0.8717525
## PR
sort(unique(paySim_PR_glm), decreasing = TRUE)[1:4]
#            type        up       down   original
# Log. Regression 0.6795277 0.02051724 0.01683785
## ROC VALUES
paySim_glm_original_ROC <- paySim_glm_results_df_roc[paySim_glm_results_df_roc$model == "up",]
paySim_glm_original_ROC$model <- "Upsampled Log. Regression"
## PR VALUES
paySim_glm_original_PR <- paySim_glm_results_df_pr[paySim_glm_results_df_pr$model == "up",]
paySim_glm_original_PR$model <- "Upsampled Log. Regression"
#### End GLM ####

#### GBM ####
paySim_auc_gbm$type <- "GBM"
paySim_PR_gbm$type <- "GBM"
## AUC
sort(unique(paySim_auc_gbm), decreasing = TRUE)[1:4]
#type        up  weighted     SMOTE
# GBM 0.9999954 0.9999931 0.9995275
## PR
sort(unique(paySim_PR_gbm), decreasing = TRUE)[1:4]
# type        up  weighted     SMOTE
# GBM 0.9983664 0.9974719 0.9108737
## ROC VALUES
paySim_gbm_original_ROC <- paySim_gbm_results_df_roc[paySim_gbm_results_df_roc$model == "weighted",]
paySim_gbm_original_ROC$model <- "Weighted GBM"
## PR VALUES
paySim_gbm_original_PR <- paySim_gbm_results_df_pr[paySim_gbm_results_df_pr$model == "weighted",]
paySim_gbm_original_PR$model <- "Weighted GBM"
#### End GBM ####

#### xGBoost ####
paySim_auc_xgboost$type <- "xGBoost"
paySim_PR_xgboost$type <- "xGBoost"
## AUC
sort(unique(paySim_auc_xgboost), decreasing = TRUE)[1:4]
#    type  original  weighted       up
# xGBoost 0.9999994 0.9999994 0.999996
## PR
sort(unique(paySim_PR_xgboost), decreasing = TRUE)[1:4]
#    type  original  weighted        up
# xGBoost 0.9998013 0.9998013 0.9985762
## ROC VALUES
paySim_xgboost_weighted_ROC <- paySim_xgb_results_df_roc[paySim_xgb_results_df_roc$model == "original",]
paySim_xgboost_weighted_ROC$model <- "Original xGBoost"
## PR VALUES
paySim_xgboost_weighted_PR <- paySim_xgb_results_df_pr[paySim_xgb_results_df_pr$model == "original",]
paySim_xgboost_weighted_PR$model <- "Original xGBoost"
#### End xGBoost ####

#### SVM ####
paySim_auc_svm$type <- "SVM"
paySim_PR_svm$type <- "SVM"
## AUC
sort(unique(paySim_auc_svm), decreasing = TRUE)[1:8]
# type       up original  weighted     SMOTE    radial weighted_rad      down
#  SVM 0.968656 0.956093 0.9553215 0.9004314 0.8788596    0.8767621 0.8765614
## PR
sort(unique(paySim_PR_svm), decreasing = TRUE)[1:8]
# type        up  weighted original     SMOTE      down     radial weighted_rad
#  SVM 0.7623474 0.7026728 0.701313 0.5677162 0.5328521 0.05113294    0.0493418

## ROC VALUES
paySim_svm_weighted_rad_ROC <- paySim_svm_results_df_roc[paySim_svm_results_df_roc$model == "up",]
paySim_svm_weighted_rad_ROC$model <- "Upsampled SVM"
## PR VALUES
paySim_svm_weighted_rad_PR <- paySim_svm_results_df_pr[paySim_svm_results_df_pr$model == "up",]
paySim_svm_weighted_rad_PR$model <- "Upsampled SVM"
#### End SVM ####

#### NNET ####
paySim_auc_nnet$type <- "ANN"
paySim_PR_nnet$type <- "ANN"
## AUC
sort(unique(paySim_auc_nnet), decreasing = TRUE)[1:4]
# type  original     SMOTE  weighted
#  ANN 0.9968546 0.9902082 0.9797525
## PR
sort(unique(paySim_PR_nnet), decreasing = TRUE)[1:4]
# type  original  weighted      down
#  ANN 0.9548565 0.9452027 0.5943965
## ROC VALUES
paySim_nnet_original_ROC <- paySim_nnet_results_df_roc[paySim_nnet_results_df_roc$model == "original",]
paySim_nnet_original_ROC$model <- "Original ANN"
## PR VALUES
paySim_nnet_original_PR <- paySim_nnet_results_df_pr[paySim_nnet_results_df_pr$model == "original",]
paySim_nnet_original_PR$model <- "Original ANN"
#### End NNET ####


#### Combining to create Graphics #### 
paySim_ROC_all <- rbind(paySim_randfor_original_ROC, paySim_glm_original_ROC,
                         paySim_gbm_original_ROC, paySim_xgboost_weighted_ROC,
                         paySim_svm_weighted_rad_ROC, paySim_nnet_original_ROC)

custom_col <- c("#000000", "red","#009E73", "#0072B2", "#D55e00", "#CC79A7")

roc_curve <- ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_ROC_all) +
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

paySim_PR_all <- rbind(paySim_randfor_original_PR, paySim_glm_original_PR,
                        paySim_gbm_original_PR, paySim_xgboost_weighted_PR,
                        paySim_svm_weighted_rad_PR, paySim_nnet_original_PR)

pr_curve <- ggplot(aes(x = recall, y = precision, group = model), data = paySim_PR_all) +
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
auc_table_all <- data.frame(sort(unique(paySim_auc_randfor), decreasing = TRUE)[2],
                            sort(unique(paysim_auc_glm), decreasing = TRUE)[2],
                            sort(unique(paySim_auc_gbm), decreasing = TRUE)[3],
                            sort(unique(paySim_auc_xgboost), decreasing = TRUE)[2],
                            sort(unique(paySim_auc_svm), decreasing = TRUE)[2],
                            sort(unique(paySim_auc_nnet), decreasing = TRUE)[2])
colnames(auc_table_all) <- c("Original Random Forest", "Upsampled GLM",
                             "Weighted GBM", "Original xGBoost",
                             "UpsampledSVM", "Original ANN")


#### Tables ####
paysim_auc_glm <- paysim_auc_glm %>%
  mutate(weighted = " ") %>%
  select(original, weighted, down, up, SMOTE, type)
paySim_auc_svm <- paySim_auc_svm %>%
  dplyr::rename(weighted_radial = weighted_rad,
                original_radial = radial) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
paySim_auc_table <- rbind(paySim_auc_randfor, paySim_auc_xgboost, 
                           paySim_auc_gbm, paySim_auc_nnet,
                           paysim_auc_glm) %>%
  mutate(weighted_radial = " ",
         original_radial = " ") %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(paySim_auc_svm) 


paySim_PR_glm <- paySim_PR_glm %>%
  mutate(weighted = " ") %>%
  select(original, weighted, down, up, SMOTE, type)
paySim_PR_svm <- paySim_PR_svm %>%
  dplyr::rename(weighted_radial = weighted_rad,
                original_radial = radial) %>%
  select(type, original, original_radial,
         weighted, weighted_radial,
         down, up, SMOTE)
paySim_PR_table <- rbind(paySim_PR_randfor, paySim_PR_xgboost, 
                          paySim_PR_gbm, paySim_PR_nnet,
                          paySim_PR_glm) %>%
  mutate(weighted_radial = " ",
         original_radial = " ") %>%
  select(type, original, original_radial, weighted, weighted_radial, down, up, SMOTE) %>%
  rbind(paySim_PR_svm) 
