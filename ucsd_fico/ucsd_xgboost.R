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
library(xgboost)
options(scipen=999)

set.seed(48)

# Loading the features and the classes
ucsd_data<- read.table(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Inputs",
                       header = TRUE,
                       sep = ",",
                       stringsAsFactors = TRUE)
ucsd_data_targets <- read.table(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Targets",
                                #header = TRUE,
                                sep = ",")
# Binding classes and features in one set
ucsd_data <- cbind(ucsd_data, ucsd_data_targets)
rm(ucsd_data_targets)

ucsd_data <- ucsd_data %>%
  dplyr::select(-c(custAttr2, total, hour2, state1)) %>%
  dplyr::rename(Class = V1)

ucsd_data$Class <- as.factor(ucsd_data$Class)
prop.table(table(ucsd_data$Class))
# 0       1 
# 0.97346 0.02654 

# Getting only those customers that appear more than one
ucsd_data <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1) %>%
  dplyr::inner_join(ucsd_data, by = "custAttr1") %>%
  dplyr::select(-freq)


# Splitting into train and test datasets
split = sample.split(ucsd_data$Class, SplitRatio = 0.6)
ucsd_train <- subset(ucsd_data, split == TRUE)
ucsd_test <- subset(ucsd_data, split == FALSE)
# ucsd_train$zip1 <- as.factor(ucsd_train$zip1)
# ucsd_test$zip1 <- as.factor(ucsd_test$zip1)
# ucsd_train$custAttr1 <- as.factor(ucsd_train$custAttr1)
# ucsd_test$custAttr1 <- as.factor(ucsd_test$custAttr1)

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
                          repeats = 1,
                          summaryFunction = twoClassSummary,
                          #allowParallel = TRUE,
                          classProbs = TRUE,
                          verboseIter = TRUE
                          )

ptm <- proc.time()
cluster <- makeCluster(detectCores() - 3) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_xgboost <- train(Class ~ .,
                      data = ucsd_train,
                      method = "xgbTree",
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)

stopCluster(cluster)
registerDoSEQ()
proc.time() - ptm
# No parallel computing
# user  system elapsed 
# 939.05   37.31  388.53 

######################################### XGBOOST PREDICTIONS
xgboost_results <- predict(ucsd_xgboost, newdata = ucsd_test)
conf_matr_xgboost <- confusionMatrix(xgboost_results, ucsd_test$Class)
conf_matr_xgboost
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15861   371
# X2    28   107
# 
# Accuracy : 0.9756               
# 95% CI : (0.9731, 0.9779)     
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.00009176           
# 
# Kappa : 0.3406               
# Mcnemar's Test P-Value : < 0.00000000000000022
# 
# Sensitivity : 0.9982               
# Specificity : 0.2238               
# Pos Pred Value : 0.9771               
# Neg Pred Value : 0.7926               
# Prevalence : 0.9708               
# Detection Rate : 0.9691               
# Detection Prevalence : 0.9918               
# Balanced Accuracy : 0.6110               
# 
# 'Positive' Class : X1          

xgboost_results_prob <- predict(ucsd_xgboost, newdata = ucsd_test, type = "prob")
xgboost_results_probs <- ifelse(xgboost_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2 <- confusionMatrix(xgboost_results_probs, ucsd_test$Class)
conf_matr_xgboost2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15403   217
# X2   486   261
# 
# Accuracy : 0.957              
# 95% CI : (0.9538, 0.9601)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.4049             
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.9694             
#             Specificity : 0.5460             
#          Pos Pred Value : 0.9861             
#          Neg Pred Value : 0.3494             
#              Prevalence : 0.9708             
#          Detection Rate : 0.9411             
#    Detection Prevalence : 0.9544             
#       Balanced Accuracy : 0.7577             
#                                              
#        'Positive' Class : X1      

trellis.par.set(caretTheme())
train_plot_xgboost <- plot(ucsd_xgboost, metric = "ROC")

xgboost_imp <- varImp(ucsd_xgboost, scale = FALSE)
plot(xgboost_imp)

# XGBOOST ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

ucsd_xgboost %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.8579

plot(roc(ucsd_test$Class, predict(ucsd_xgboost, ucsd_test,type = "prob")[,"X2"]))


############################### COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
ucsd_model_weights <- ifelse(ucsd_train$Class == "X1",
                               (1/table(ucsd_train$Class)[1]) * 0.5,
                               (1/table(ucsd_train$Class)[2]) * 0.5)

ctrl_ucsd$seeds <- ucsd_xgboost$control$seeds

cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
ucsd_xgboost_weighted_fit <- train(Class ~ .,
                                   data = ucsd_train,
                                   method = "xgbTree",
                                   verbose = FALSE,
                                   weights = ucsd_model_weights,
                                   metric = "ROC", 
                                   trControl = ctrl_ucsd)

stopCluster(cluster)
registerDoSEQ()

xgboost_results_weight <- predict(ucsd_xgboost_weighted_fit, newdata = ucsd_test)
conf_matr_xgboost_weight <- confusionMatrix(xgboost_results_weight, ucsd_test$Class)
conf_matr_xgboost_weight

xgboost_results_prob_weight <- predict(ucsd_xgboost_weighted_fit, newdata = ucsd_test, type = "prob")
xgboost_results_probs_weight <- ifelse(xgboost_results_prob_weight$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2_weight <- confusionMatrix(xgboost_results_probs_weight, ucsd_test$Class)
conf_matr_xgboost2_weight

trellis.par.set(caretTheme())
train_plot_xgboost_weight <- plot(ucsd_xgboost_weighted_fit, metric = "ROC")

xgboost_imp_weight <- varImp(ucsd_xgboost_weighted_fit, scale = FALSE)
plot(xgboost_imp_weight)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"
cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)
ucsd_xgboost_down_fit <- train(Class ~ .,
                                 data = ucsd_train,
                                 method = "xgbTree",
                                 verbose = FALSE,
                                 metric = "ROC",
                                 trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

xgboost_results_down <- predict(ucsd_xgboost_down_fit, newdata = ucsd_test)
conf_matr_xgboost_down <- confusionMatrix(xgboost_results_down, ucsd_test$Class)
conf_matr_xgboost_down
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 11613   105
# X2  4276   373
# 
# Accuracy : 0.7323             
# 95% CI : (0.7255, 0.7391)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0977             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.73088            
# Specificity : 0.78033            
# Pos Pred Value : 0.99104            
# Neg Pred Value : 0.08023            
# Prevalence : 0.97079            
# Detection Rate : 0.70954            
# Detection Prevalence : 0.71595            
# Balanced Accuracy : 0.75561            
# 
# 'Positive' Class : X1  

xgboost_results_prob_down <- predict(ucsd_xgboost_down_fit, newdata = ucsd_test, type = "prob")
xgboost_results_probs_down <- ifelse(xgboost_results_prob_down$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2_down <- confusionMatrix(xgboost_results_probs_down, ucsd_test$Class)
conf_matr_xgboost2_down
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1  3494    16
# X2 12395   462
# 
# Accuracy : 0.2417             
# 95% CI : (0.2352, 0.2483)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0137             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.21990            
# Specificity : 0.96653            
# Pos Pred Value : 0.99544            
# Neg Pred Value : 0.03593            
# Prevalence : 0.97079            
# Detection Rate : 0.21348            
# Detection Prevalence : 0.21446            
# Balanced Accuracy : 0.59321            
# 
# 'Positive' Class : X1    

trellis.par.set(caretTheme())
train_plot_xgboost_down <- plot(ucsd_xgboost_down_fit, metric = "ROC")

xgboost_imp_down <- varImp(ucsd_xgboost_down_fit, scale = FALSE)
plot(xgboost_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_xgboost_up_fit <- train(Class ~ .,
                               data = ucsd_train,
                               method = "xgbTree",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

xgboost_results_up <- predict(ucsd_xgboost_up_fit, newdata = ucsd_test)
conf_matr_xgboost_up <- confusionMatrix(xgboost_results_up, ucsd_test$Class)
conf_matr_xgboost_up
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 14167   139
# X2  1722   339
# 
# Accuracy : 0.8863             
# 95% CI : (0.8813, 0.8911)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.2306             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.8916             
# Specificity : 0.7092             
# Pos Pred Value : 0.9903             
# Neg Pred Value : 0.1645             
# Prevalence : 0.9708             
# Detection Rate : 0.8656             
# Detection Prevalence : 0.8741             
# Balanced Accuracy : 0.8004             
#
#'Positive' Class : X1  

xgboost_results_prob_up <- predict(ucsd_xgboost_up_fit, newdata = ucsd_test, type = "prob")
xgboost_results_probs_up <- ifelse(xgboost_results_prob_up$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2_up <- confusionMatrix(xgboost_results_probs_up, ucsd_test$Class)
conf_matr_xgboost2_up
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   X1   X2
# X1 6587   42
# X2 9302  436
# 
# Accuracy : 0.4291             
# 95% CI : (0.4215, 0.4367)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0314             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.41456            
# Specificity : 0.91213            
# Pos Pred Value : 0.99366            
# Neg Pred Value : 0.04477            
# Prevalence : 0.97079            
# Detection Rate : 0.40246            
# Detection Prevalence : 0.40502            
# Balanced Accuracy : 0.66335            
# 
# 'Positive' Class : X1  
trellis.par.set(caretTheme())
train_plot_xgboost_up <- plot(ucsd_xgboost_up_fit, metric = "ROC")

xgboost_imp_up <- varImp(ucsd_xgboost_up_fit, scale = FALSE)
plot(xgboost_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_xgboost_smote_fit <- train(Class ~ .,
                                  data = ucsd_train,
                                  method = "xgbTree",
                                  verbose = FALSE,
                                  metric = "ROC",
                                  trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

xgboost_results_smote <- predict(ucsd_xgboost_smote_fit, newdata = ucsd_test)
conf_matr_xgboost_smote <- confusionMatrix(xgboost_results_smote, ucsd_test$Class)
conf_matr_xgboost_smote
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 14879   227
# X2  1010   251
# 
# Accuracy : 0.9244             
# 95% CI : (0.9203, 0.9284)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.2572             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9364             
# Specificity : 0.5251             
# Pos Pred Value : 0.9850             
# Neg Pred Value : 0.1990             
# Prevalence : 0.9708             
# Detection Rate : 0.9091             
# Detection Prevalence : 0.9230             
# Balanced Accuracy : 0.7308             
# 
# 'Positive' Class : X1                 

xgboost_results_prob_smote <- predict(ucsd_xgboost_smote_fit, newdata = ucsd_test, type = "prob")
xgboost_results_probs_smote <- ifelse(xgboost_results_prob_smote$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2_smote <- confusionMatrix(xgboost_results_probs_smote, ucsd_test$Class)
conf_matr_xgboost2_smote
# Confusion Matrix and Statistics
# 
# Reference
# Prediction   X1   X2
# X1 7614   54
# X2 8275  424
# 
# Accuracy : 0.4911             
# 95% CI : (0.4834, 0.4988)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0392             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.47920            
# Specificity : 0.88703            
# Pos Pred Value : 0.99296            
# Neg Pred Value : 0.04874            
# Prevalence : 0.97079            
# Detection Rate : 0.46520            
# Detection Prevalence : 0.46850            
# Balanced Accuracy : 0.68311            
# 
# 'Positive' Class : X1 
trellis.par.set(caretTheme())
train_plot_xgboost_smote <- plot(ucsd_xgboost_smote_fit, metric = "ROC")

xgboost_imp_smote <- varImp(ucsd_xgboost_smote_fit, scale = FALSE)
plot(xgboost_imp_smote)


####################################################################

ucsd_xgboost_model_list <- list(original = ucsd_xgboost,
                              weighted = ucsd_xgboost_weighted_fit,
                              down = ucsd_xgboost_down_fit,
                              up = ucsd_xgboost_up_fit,
                              SMOTE = ucsd_xgboost_smote_fit)


ucsd_xgboost_model_list_roc <- ucsd_xgboost_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_xgboost_model_list_roc %>%
  map(auc)
# $original
# Area under the curve: 0.8579
# 
# $weighted
# Area under the curve: 0.5
# 
# $down
# Area under the curve: 0.8201
# 
# $up
# Area under the curve: 0.8576
# 
# $SMOTE
# Area under the curve: 0.8357

ucsd_xgboost_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in ucsd_xgboost_model_list_roc){
  ucsd_xgboost_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(ucsd_xgboost_model_list)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_xgboost_results_df_roc <- bind_rows(ucsd_xgboost_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_xgboost_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
ucsd_xgboost_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

ucsd_xgboost_model_list_pr <- ucsd_xgboost_model_list %>%
  map(ucsd_xgboost_calc_auprc, data = ucsd_test)

# Precision recall Curve AUC calculation
ucsd_xgboost_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)
# $original
# [1] 0.517209
# 
# $weighted
# [1] 0.9974719
# 
# $down
# [1] 0.6088192
# 
# $up
# [1] 0.9983664
# 
# $SMOTE
# [1] 0.9759452


ucsd_xgboost_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in ucsd_xgboost_model_list_pr) {
  ucsd_xgboost_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(ucsd_xgboost_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_xgboost_results_df_pr <- bind_rows(ucsd_xgboost_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = ucsd_xgboost_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(ucsd_test$type == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)

#####################################################################################################
ucsd_xgboostSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  the_curve <- pr.curve(data$X2[index_class2],
                        data$X2[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}