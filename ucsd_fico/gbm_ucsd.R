library(data.table)
library(dplyr)
library(caret)
library(DMwR) #SMOTE
library(purrr)
library(pROC)
library(PRROC)
library(caTools)
library(doParallel)
library(parallel)
library(plyr)
library(gbm)
options(scipen=999)

set.seed(48)

ucsd_data<- read.table(file = "C:/Users/zxmum28/Documents/MT/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Inputs",
                       header = TRUE,
                       sep = ",",
                       stringsAsFactors = TRUE)
ucsd_data_targets <- read.table(file = "C:/Users/zxmum28/Documents/MT/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Targets",
                                #header = TRUE,
                                sep = ",")
ucsd_data <- cbind(ucsd_data, ucsd_data_targets)
rm(ucsd_data_targets)

ucsd_data <- ucsd_data %>%
  dplyr::select(-c(custAttr2, total, hour2, state1)) %>%
  dplyr::rename(Class = V1)

ucsd_data$Class <- as.factor(ucsd_data$Class)
prop.table(table(ucsd_data$Class))
# 0       1 
# 0.97346 0.02654 

multi_obs <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1)

ucsd_data <- join(ucsd_data, multi_obs, by = "custAttr1", type = "inner") %>%
  dplyr::select(-freq)
rm(multi_obs)

split = sample.split(ucsd_data$Class, SplitRatio = 0.6)
ucsd_train <- subset(ucsd_data, split == TRUE)
ucsd_test <- subset(ucsd_data, split == FALSE)

feature.names=names(ucsd_train)
for (f in feature.names) {
  if (class(ucsd_train[[f]])=="factor") {
    levels <- unique(c(ucsd_train[[f]]))
    ucsd_train[[f]] <- factor(ucsd_train[[f]],
                              labels=make.names(levels))
  }
}
feature.names2=names(ucsd_test)
for (f in feature.names2) {
  if (class(ucsd_test[[f]])=="factor") {
    levels <- unique(c(ucsd_test[[f]]))
    ucsd_test[[f]] <- factor(ucsd_test[[f]],
                             labels=make.names(levels))
  }
}

rm(ucsd_data)

prop.table(table(ucsd_train$Class))
# X1         X2 
# 0.97075476 0.02924524 
prop.table(table(ucsd_test$Class))
# X1         X2 
# 0.97079489 0.02920511 

ctrl_ucsd <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 2,
                          summaryFunction = twoClassSummary,
                          #allowParallel = TRUE,
                          classProbs = TRUE,
                          verboseIter = TRUE
)

cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_gbm <- train(Class ~ .,
                      data = ucsd_train,
                      method = "gbm",
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

######################################### XGBOOST PREDICTIONS
gbm_results <- predict(ucsd_gbm, newdata = ucsd_test)
conf_matr_gbm <- confusionMatrix(gbm_results, ucsd_test$Class)
conf_matr_gbm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 14346   291
# X2  1543   187
# 
# Accuracy : 0.8879             
# 95% CI : (0.883, 0.8927)    
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.1295             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9029             
# Specificity : 0.3912             
# Pos Pred Value : 0.9801             
# Neg Pred Value : 0.1081             
# Prevalence : 0.9708             
# Detection Rate : 0.8765             
# Detection Prevalence : 0.8943             
# Balanced Accuracy : 0.6471             
# 
# 'Positive' Class : X1       

gbm_results_prob <- predict(ucsd_gbm, newdata = ucsd_test, type = "prob")
gbm_results_probs <- ifelse(gbm_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_gbm2 <- confusionMatrix(gbm_results_probs, ucsd_test$Class)
conf_matr_gbm2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1  1735     9
# X2 14154   469
# 
# Accuracy : 0.1347             
# 95% CI : (0.1295, 0.14)     
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0059             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.10920            
# Specificity : 0.98117            
# Pos Pred Value : 0.99484            
# Neg Pred Value : 0.03207            
# Prevalence : 0.97079            
# Detection Rate : 0.10601            
# Detection Prevalence : 0.10656            
# Balanced Accuracy : 0.54518            
# 
# 'Positive' Class : X1         

trellis.par.set(caretTheme())
train_plot_gbm <- plot(ucsd_gbm, metric = "ROC")

gbm_imp <- varImp(ucsd_gbm)
plot(gbm_imp)

# XGBOOST ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

ucsd_gbm %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.7584

plot(roc(ucsd_test$Class, predict(ucsd_gbm, ucsd_test,type = "prob")[,"X2"]))


############################### COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
ucsd_model_weights <- ifelse(ucsd_train$Class == "X1",
                             (1/table(ucsd_train$Class)[1]) * 0.5,
                             (1/table(ucsd_train$Class)[2]) * 0.5)

ctrl_ucsd$seeds <- ucsd_gbm$control$seeds

cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
ucsd_gbm_weighted_fit <- train(Class ~ .,
                                   data = ucsd_train,
                                   method = "gbm",
                                   verbose = FALSE,
                                   weights = ucsd_model_weights,
                                   metric = "ROC", 
                                   trControl = ctrl_ucsd)

stopCluster(cluster)
registerDoSEQ()

gbm_results_weight <- predict(ucsd_gbm_weighted_fit, newdata = ucsd_test)
conf_matr_gbm_weight <- confusionMatrix(gbm_results_weight, ucsd_test$Class)
conf_matr_gbm_weight
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 13115   249
# X2  2774   229
# 
# Accuracy : 0.8153             
# 95% CI : (0.8093, 0.8212)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0855             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.82541            
# Specificity : 0.47908            
# Pos Pred Value : 0.98137            
# Neg Pred Value : 0.07626            
# Prevalence : 0.97079            
# Detection Rate : 0.80131            
# Detection Prevalence : 0.81652            
# Balanced Accuracy : 0.65225            
# 
# 'Positive' Class : X1  

gbm_results_prob_weight <- predict(ucsd_gbm_weighted_fit, newdata = ucsd_test, type = "prob")
gbm_results_probs_weight <- ifelse(gbm_results_prob_weight$X2 > 0.1, "X2", "X1")
conf_matr_gbm2_weight <- confusionMatrix(gbm_results_probs_weight, ucsd_test$Class)
conf_matr_gbm2_weight

trellis.par.set(caretTheme())
train_plot_gbm_weight <- plot(ucsd_gbm_weighted_fit, metric = "ROC")

gbm_imp_weight <- varImp(ucsd_gbm_weighted_fit, scale = FALSE)
plot(gbm_imp_weight)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"
cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)
ucsd_gbm_down_fit <- train(Class ~ .,
                               data = ucsd_train,
                               method = "gbm",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

gbm_results_down <- predict(ucsd_gbm_down_fit, newdata = ucsd_test)
conf_matr_gbm_down <- confusionMatrix(gbm_results_down, ucsd_test$Class)
conf_matr_gbm_down
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 10946   130
# X2  4943   348
# 
# Accuracy : 0.69               
# 95% CI : (0.6829, 0.6971)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0709             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.68890            
# Specificity : 0.72803            
# Pos Pred Value : 0.98826            
# Neg Pred Value : 0.06577            
# Prevalence : 0.97079            
# Detection Rate : 0.66878            
# Detection Prevalence : 0.67673            
# Balanced Accuracy : 0.70847            
# 
# 'Positive' Class : X1  
gbm_results_prob_down <- predict(ucsd_gbm_down_fit, newdata = ucsd_test, type = "prob")
gbm_results_probs_down <- ifelse(gbm_results_prob_down$X2 > 0.1, "X2", "X1")
conf_matr_gbm2_down <- confusionMatrix(gbm_results_probs_down, ucsd_test$Class)
conf_matr_gbm2_down

trellis.par.set(caretTheme())
train_plot_gbm_down <- plot(ucsd_gbm_down_fit, metric = "ROC")

gbm_imp_down <- varImp(ucsd_gbm_down_fit, scale = FALSE)
plot(gbm_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_gbm_up_fit <- train(Class ~ .,
                             data = ucsd_train,
                             method = "gbm",
                             verbose = FALSE,
                             metric = "ROC",
                             trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

gbm_results_up <- predict(ucsd_gbm_up_fit, newdata = ucsd_test)
conf_matr_gbm_up <- confusionMatrix(gbm_results_up, ucsd_test$Class)
conf_matr_gbm_up
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 11961   125
# X2  3928   353
# 
# Accuracy : 0.7524             
# 95% CI : (0.7457, 0.759)    
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.1011             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.75278            
# Specificity : 0.73849            
# Pos Pred Value : 0.98966            
# Neg Pred Value : 0.08246            
# Prevalence : 0.97079            
# Detection Rate : 0.73080            
# Detection Prevalence : 0.73844            
# Balanced Accuracy : 0.74564            
# 
# 'Positive' Class : X1 
gbm_results_prob_up <- predict(ucsd_gbm_up_fit, newdata = ucsd_test, type = "prob")
gbm_results_probs_up <- ifelse(gbm_results_prob_up$X2 > 0.1, "X2", "X1")
conf_matr_gbm2_up <- confusionMatrix(gbm_results_probs_up, ucsd_test$Class)
conf_matr_gbm2_up

trellis.par.set(caretTheme())
train_plot_gbm_up <- plot(ucsd_gbm_up_fit, metric = "ROC")

gbm_imp_up <- varImp(ucsd_gbm_up_fit, scale = FALSE)
plot(gbm_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_gbm_smote_fit <- train(Class ~ .,
                                data = ucsd_train,
                                method = "gbm",
                                verbose = FALSE,
                                metric = "ROC",
                                trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

gbm_results_smote <- predict(ucsd_gbm_smote_fit, newdata = ucsd_test)
conf_matr_gbm_smote <- confusionMatrix(gbm_results_smote, ucsd_test$Class)
conf_matr_gbm_smote
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 14346   291
# X2  1543   187
# 
# Accuracy : 0.8879             
# 95% CI : (0.883, 0.8927)    
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.1295             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9029             
# Specificity : 0.3912             
# Pos Pred Value : 0.9801             
# Neg Pred Value : 0.1081             
# Prevalence : 0.9708             
# Detection Rate : 0.8765             
# Detection Prevalence : 0.8943             
# Balanced Accuracy : 0.6471             
# 
# 'Positive' Class : X1    
gbm_results_prob_smote <- predict(ucsd_gbm_smote_fit, newdata = ucsd_test, type = "prob")
gbm_results_probs_smote <- ifelse(gbm_results_prob_smote$X2 > 0.1, "X2", "X1")
conf_matr_gbm2_smote <- confusionMatrix(gbm_results_probs_smote, ucsd_test$Class)
conf_matr_gbm2_smote

trellis.par.set(caretTheme())
train_plot_gbm_smote <- plot(ucsd_gbm_smote_fit, metric = "ROC")

gbm_imp_smote <- varImp(ucsd_gbm_smote_fit, scale = FALSE)
plot(gbm_imp_smote)


####################################################################

ucsd_gbm_model_list <- list(original = ucsd_gbm,
                                    weighted = ucsd_gbm_weighted_fit,
                                    down = ucsd_gbm_down_fit,
                                    up = ucsd_gbm_up_fit,
                                    SMOTE = ucsd_gbm_smote_fit)


ucsd_gbm_model_list_roc <- ucsd_gbm_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_gbm_model_list_roc %>%
  map(auc)
# $original
# Area under the curve: 0.7584
# 
# $weighted
# Area under the curve: 0.7279
# 
# $down
# Area under the curve: 0.7764
# 
# $up
# Area under the curve: 0.8044
# 
# $SMOTE
# Area under the curve: 0.7584

ucsd_gbm_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in ucsd_gbm_model_list_roc){
  ucsd_gbm_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(ucsd_gbm_model_list)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_gbm_results_df_roc <- bind_rows(ucsd_gbm_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_gbm_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
ucsd_gbm_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

ucsd_gbm_model_list_pr <- ucsd_gbm_model_list %>%
  map(ucsd_gbm_calc_auprc, data = ucsd_test)

# Precision recall Curve AUC calculation
ucsd_gbm_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)
# $original
# [1] 0.1321036
# 
# $weighted
# [1] 0.08574934
# 
# $down
# [1] 0.1192452
# 
# $up
# [1] 0.2112761
# 
# $SMOTE
# [1] 0.1321036


ucsd_gbm_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in ucsd_gbm_model_list_pr) {
  ucsd_gbm_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(ucsd_gbm_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_gbm_results_df_pr <- bind_rows(ucsd_gbm_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = ucsd_gbm_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(ucsd_test$type == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)

#####################################################################################################
ucsd_gbmSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  the_curve <- pr.curve(data$X2[index_class2],
                        data$X2[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}