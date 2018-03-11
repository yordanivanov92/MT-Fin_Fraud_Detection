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
library(nnet)
options(scipen=999)

set.seed(48)

ucsd_data<- read.table(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Inputs",
                       header = TRUE,
                       sep = ",",
                       stringsAsFactors = TRUE)
ucsd_data_targets <- read.table(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/UCSD-FICO competition/DataminingContest2009.Task2.Train.Targets",
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

nnet_grid <- expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), .size = c(3, 5, 10, 20))
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
ucsd_nnet <- train(Class ~ .,
                   data = ucsd_train,
                   method = "nnet",
                   linout = FALSE,
                   verbose = FALSE,
                   metric = "ROC",
                   #tuneGrid = mxnet.params,
                   trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

######################################### Randfor PREDICTIONS
nnet_results <- predict(ucsd_nnet, newdata = ucsd_test)
conf_matr_nnet <- confusionMatrix(nnet_results, ucsd_test$Class)
conf_matr_nnet
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15818   276
# X2    71   202
# 
# Accuracy : 0.9788               
# 95% CI : (0.9765, 0.981)      
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.0000000001049      
# 
# Kappa : 0.5279               
# Mcnemar's Test P-Value : < 0.00000000000000022
# 
# Sensitivity : 0.9955               
# Specificity : 0.4226               
# Pos Pred Value : 0.9829               
# Neg Pred Value : 0.7399               
# Prevalence : 0.9708               
# Detection Rate : 0.9665               
# Detection Prevalence : 0.9833               
# Balanced Accuracy : 0.7091               
# 
# 'Positive' Class : X1 

nnet_results_prob <- predict(ucsd_nnet, newdata = ucsd_test, type = "prob")
nnet_results_probs <- ifelse(nnet_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_nnet2 <- confusionMatrix(nnet_results_probs, ucsd_test$Class)
conf_matr_nnet2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15326   156
# X2   563   322
# 
# Accuracy : 0.9561             
# 95% CI : (0.9528, 0.9592)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.4517             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9646             
# Specificity : 0.6736             
# Pos Pred Value : 0.9899             
# Neg Pred Value : 0.3638             
# Prevalence : 0.9708             
# Detection Rate : 0.9364             
# Detection Prevalence : 0.9459             
# Balanced Accuracy : 0.8191             
# 
# 'Positive' Class : X1       

trellis.par.set(caretTheme())
train_plot_nnet <- plot(ucsd_nnet, metric = "ROC")

nnet_imp <- varImp(ucsd_nnet)
plot(nnet_imp)

# XGBOOST ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

ucsd_nnet %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.8872

plot(roc(ucsd_test$Class, predict(ucsd_nnet, ucsd_test,type = "prob")[,"X2"]))


############################### COST SENSITIVE RANDFOR MODEL
# The penalization costs can be tinkered with
ucsd_model_weights <- ifelse(ucsd_train$Class == "X1",
                             (1/table(ucsd_train$Class)[1]) * 0.5,
                             (1/table(ucsd_train$Class)[2]) * 0.5)

ctrl_ucsd$seeds <- ucsd_nnet$control$seeds

cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
ucsd_nnet_weighted_fit <- train(Class ~ .,
                                   data = ucsd_train,
                                   method = "nnet",
                                   verbose = FALSE,
                                   weights = ucsd_model_weights,
                                   metric = "ROC", 
                                   trControl = ctrl_ucsd)

stopCluster(cluster)
registerDoSEQ()

nnet_results_weight <- predict(ucsd_nnet_weighted_fit, newdata = ucsd_test)
conf_matr_nnet_weight <- confusionMatrix(nnet_results_weight, ucsd_test$Class)
conf_matr_nnet_weight
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15818   276
# X2    71   202
# 
# Accuracy : 0.9788               
# 95% CI : (0.9765, 0.981)      
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.0000000001049      
# 
# Kappa : 0.5279               
# Mcnemar's Test P-Value : < 0.00000000000000022
# 
# Sensitivity : 0.9955               
# Specificity : 0.4226               
# Pos Pred Value : 0.9829               
# Neg Pred Value : 0.7399               
# Prevalence : 0.9708               
# Detection Rate : 0.9665               
# Detection Prevalence : 0.9833               
# Balanced Accuracy : 0.7091               
# 
# 'Positive' Class : X1  

nnet_results_prob_weight <- predict(ucsd_nnet_weighted_fit, newdata = ucsd_test, type = "prob")
nnet_results_probs_weight <- ifelse(nnet_results_prob_weight$X2 > 0.1, "X2", "X1")
conf_matr_nnet2_weight <- confusionMatrix(nnet_results_probs_weight, ucsd_test$Class)
conf_matr_nnet2_weight
# same as with no weights

trellis.par.set(caretTheme())
train_plot_nnet_weight <- plot(ucsd_nnet_weighted_fit, metric = "ROC")

nnet_imp_weight <- varImp(ucsd_nnet_weighted_fit, scale = FALSE)
plot(nnet_imp_weight)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"
cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)
ucsd_nnet_down_fit <- train(Class ~ .,
                               data = ucsd_train,
                               method = "nnet",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

nnet_results_down <- predict(ucsd_nnet_down_fit, newdata = ucsd_test)
conf_matr_nnet_down <- confusionMatrix(nnet_results_down, ucsd_test$Class)
conf_matr_nnet_down
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 12189   106
# X2  3700   372
# 
# Accuracy : 0.7675             
# 95% CI : (0.7609, 0.7739)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.1174             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.76713            
# Specificity : 0.77824            
# Pos Pred Value : 0.99138            
# Neg Pred Value : 0.09136            
# Prevalence : 0.97079            
# Detection Rate : 0.74473            
# Detection Prevalence : 0.75121            
# Balanced Accuracy : 0.77269            
# 
# 'Positive' Class : X1 
nnet_results_prob_down <- predict(ucsd_nnet_down_fit, newdata = ucsd_test, type = "prob")
nnet_results_probs_down <- ifelse(nnet_results_prob_down$X2 > 0.1, "X2", "X1")
conf_matr_nnet2_down <- confusionMatrix(nnet_results_probs_down, ucsd_test$Class)
conf_matr_nnet2_down
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1   714     2
# X2 15175   476
# 
# Accuracy : 0.0727             
# 95% CI : (0.0688, 0.0768)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0025             
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.04494            
#             Specificity : 0.99582            
#          Pos Pred Value : 0.99721            
#          Neg Pred Value : 0.03041            
#              Prevalence : 0.97079            
#          Detection Rate : 0.04362            
#    Detection Prevalence : 0.04375            
#       Balanced Accuracy : 0.52038            
#                                              
#        'Positive' Class : X1     

trellis.par.set(caretTheme())
train_plot_nnet_down <- plot(ucsd_nnet_down_fit, metric = "ROC")

nnet_imp_down <- varImp(ucsd_nnet_down_fit, scale = FALSE)
plot(nnet_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_nnet_up_fit <- train(Class ~ .,
                             data = ucsd_train,
                             method = "nnet",
                             verbose = FALSE,
                             metric = "ROC",
                             trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

nnet_results_up <- predict(ucsd_nnet_up_fit, newdata = ucsd_test)
conf_matr_nnet_up <- confusionMatrix(nnet_results_up, ucsd_test$Class)
conf_matr_nnet_up
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15797   248
# X2    92   230
# 
# Accuracy : 0.9792               
# 95% CI : (0.9769, 0.9814)     
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.000000000009406    
# 
# Kappa : 0.5648               
# Mcnemar's Test P-Value : < 0.00000000000000022
# 
# Sensitivity : 0.9942               
# Specificity : 0.4812               
# Pos Pred Value : 0.9845               
# Neg Pred Value : 0.7143               
# Prevalence : 0.9708               
# Detection Rate : 0.9652               
# Detection Prevalence : 0.9803               
# Balanced Accuracy : 0.7377               
# 
# 'Positive' Class : X1  
nnet_results_prob_up <- predict(ucsd_nnet_up_fit, newdata = ucsd_test, type = "prob")
nnet_results_probs_up <- ifelse(nnet_results_prob_up$X2 > 0.1, "X2", "X1")
conf_matr_nnet2_up <- confusionMatrix(nnet_results_probs_up, ucsd_test$Class)
conf_matr_nnet2_up
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15059   155
# X2   830   323
# 
# Accuracy : 0.9398             
# 95% CI : (0.9361, 0.9434)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.3701             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9478             
# Specificity : 0.6757             
# Pos Pred Value : 0.9898             
# Neg Pred Value : 0.2801             
# Prevalence : 0.9708             
# Detection Rate : 0.9201             
# Detection Prevalence : 0.9296             
# Balanced Accuracy : 0.8117             
# 
# 'Positive' Class : X1 
trellis.par.set(caretTheme())
train_plot_nnet_up <- plot(ucsd_nnet_up_fit, metric = "ROC")

nnet_imp_up <- varImp(ucsd_nnet_up_fit, scale = FALSE)
plot(nnet_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_nnet_smote_fit <- train(Class ~ .,
                                data = ucsd_train,
                                method = "nnet",
                                verbose = FALSE,
                                metric = "ROC",
                                trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

nnet_results_smote <- predict(ucsd_nnet_smote_fit, newdata = ucsd_test)
conf_matr_nnet_smote <- confusionMatrix(nnet_results_smote, ucsd_test$Class)
conf_matr_nnet_smote
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 14998   214
# X2   891   264
# 
# Accuracy : 0.9325             
# 95% CI : (0.9285, 0.9363)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.2942             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9439             
# Specificity : 0.5523             
# Pos Pred Value : 0.9859             
# Neg Pred Value : 0.2286             
# Prevalence : 0.9708             
# Detection Rate : 0.9164             
# Detection Prevalence : 0.9294             
# Balanced Accuracy : 0.7481             
# 
# 'Positive' Class : X1    
nnet_results_prob_smote <- predict(ucsd_nnet_smote_fit, newdata = ucsd_test, type = "prob")
nnet_results_probs_smote <- ifelse(nnet_results_prob_smote$X2 > 0.1, "X2", "X1")
conf_matr_nnet2_smote <- confusionMatrix(nnet_results_probs_smote, ucsd_test$Class)
conf_matr_nnet2_smote
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1  4432    19
# X2 11457   459
# 
# Accuracy : 0.2988             
# 95% CI : (0.2918, 0.3059)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.019              
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.27894            
# Specificity : 0.96025            
# Pos Pred Value : 0.99573            
# Neg Pred Value : 0.03852            
# Prevalence : 0.97079            
# Detection Rate : 0.27079            
# Detection Prevalence : 0.27195            
# Balanced Accuracy : 0.61959            
# 
# 'Positive' Class : X1   

trellis.par.set(caretTheme())
train_plot_nnet_smote <- plot(ucsd_nnet_smote_fit, metric = "ROC")

nnet_imp_smote <- varImp(ucsd_nnet_smote_fit, scale = FALSE)
plot(nnet_imp_smote)


####################################################################

ucsd_nnet_model_list <- list(original = ucsd_nnet,
                                weighted = ucsd_nnet_weighted_fit,
                                down = ucsd_nnet_down_fit,
                                up = ucsd_nnet_up_fit,
                                SMOTE = ucsd_nnet_smote_fit)


ucsd_nnet_model_list_roc <- ucsd_nnet_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_nnet_model_list_roc %>%
  map(auc)
# $original
# Area under the curve: 0.8872
# 
# $weighted
# Area under the curve: 0.8872
# 
# $down
# Area under the curve: 0.8596
# 
# $up
# Area under the curve: 0.8841
# 
# $SMOTE
# Area under the curve: 0.8505

ucsd_nnet_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in ucsd_nnet_model_list_roc){
  ucsd_nnet_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(ucsd_nnet_model_list)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_nnet_results_df_roc <- bind_rows(ucsd_nnet_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_nnet_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
ucsd_nnet_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

ucsd_nnet_model_list_pr <- ucsd_nnet_model_list %>%
  map(ucsd_nnet_calc_auprc, data = ucsd_test)

# Precision recall Curve AUC calculation
ucsd_nnet_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)
# $original
# [1] 0.5387054
# 
# $weighted
# [1] 0.5387054
# 
# $down
# [1] 0.4238394
# 
# $up
# [1] 0.4787437
# 
# $SMOTE
# [1] 0.3949578


ucsd_nnet_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in ucsd_nnet_model_list_pr) {
  ucsd_nnet_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(ucsd_nnet_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_nnet_results_df_pr <- bind_rows(ucsd_nnet_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = ucsd_nnet_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(ucsd_test$type == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)

#####################################################################################################
ucsd_nnetSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  the_curve <- pr.curve(data$X2[index_class2],
                        data$X2[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}