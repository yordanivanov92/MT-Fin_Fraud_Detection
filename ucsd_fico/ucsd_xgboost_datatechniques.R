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

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_xgboost <- train(Class ~ .,
                      data = ucsd_train,
                      method = "xgbTree",
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

# XGBOOST PREDICTIONS
xgboost_results <- predict(ucsd_xgboost, newdata = ucsd_test)
conf_matr_xgboost <- confusionMatrix(xgboost_results, ucsd_test$Class)
conf_matr_xgboost
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15861   385
# X2    28    93
# 
# Accuracy : 0.9748               
# 95% CI : (0.9722, 0.9771)     
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.001121             
# 
# Kappa : 0.3023               
# Mcnemar's Test P-Value : < 0.00000000000000022
#                                                
#             Sensitivity : 0.9982               
#             Specificity : 0.1946               
#          Pos Pred Value : 0.9763               
#          Neg Pred Value : 0.7686               
#              Prevalence : 0.9708               
#          Detection Rate : 0.9691               
#    Detection Prevalence : 0.9926               
#       Balanced Accuracy : 0.5964               
#                                                
#        'Positive' Class : X1             

xgboost_results_prob <- predict(ucsd_xgboost, newdata = ucsd_test, type = "prob")
xgboost_results_probs <- ifelse(xgboost_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_xgboost2 <- confusionMatrix(xgboost_results_probs, ucsd_test$Class)
conf_matr_xgboost2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15388   213
# X2   501   265
# 
# Accuracy : 0.9564             
# 95% CI : (0.9531, 0.9595)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.4046             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9685             
# Specificity : 0.5544             
# Pos Pred Value : 0.9863             
# Neg Pred Value : 0.3460             
# Prevalence : 0.9708             
# Detection Rate : 0.9402             
# Detection Prevalence : 0.9532             
# Balanced Accuracy : 0.7614             
# 
# 'Positive' Class : X1           

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

### Logistic Regression
ucsd_logistc <- train(Class ~ .,
                      data = ucsd_train,
                      method = "glm",
                      family = "binomial",
                      metric = "ROC", 
                      trControl = ctrl_ucsd)

logistic_results <- predict(ucsd_logistc, newdata = ucsd_test)
conf_matr_logistic <- confusionMatrix(logistic_results, ucsd_test$Class)
conf_matr_logistic
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15888   477
# X2     1     1
# 
# Accuracy : 0.9708             
# 95% CI : (0.9681, 0.9733)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.5122             
# 
# Kappa : 0.0039             
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.999937           
#             Specificity : 0.002092           
#          Pos Pred Value : 0.970852           
#          Neg Pred Value : 0.500000           
#              Prevalence : 0.970795           
#          Detection Rate : 0.970734           
#    Detection Prevalence : 0.999878           
#       Balanced Accuracy : 0.501015           
#                                              
#        'Positive' Class : X1  

logistic_results_prob <- predict(ucsd_logistc, newdata = ucsd_test, type = "prob")
logistic_results_probs <- ifelse(logistic_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_logistic2 <- confusionMatrix(logistic_results_probs, ucsd_test$Class)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15795   457
# X2    94    21
# 
# Accuracy : 0.9663             
# 95% CI : (0.9635, 0.969)    
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.9996             
# 
# Kappa : 0.0602             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.99408            
# Specificity : 0.04393            
# Pos Pred Value : 0.97188            
# Neg Pred Value : 0.18261            
# Prevalence : 0.97079            
# Detection Rate : 0.96505            
# Detection Prevalence : 0.99297            
# Balanced Accuracy : 0.51901            
# 
# 'Positive' Class : X1               

ucsd_logistc %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.7103

plot(roc(ucsd_test$Class, predict(ucsd_logistc, ucsd_test,type = "prob")[,"X2"]))

summary(ucsd_logistc)
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.2969  -0.2705  -0.2116  -0.1617   3.2968  
# 
# Coefficients:
#   Estimate           Std. Error z value             Pr(>|z|)    
# (Intercept) 7549558296.433304787 2395097474.169583797   3.152              0.00162 ** 
# amount              -0.004680726          0.003261308  -1.435              0.15122    
# hour1               -0.007679865          0.006763629  -1.135              0.25618    
# zip1                 0.000083745          0.000121675   0.688              0.49128    
# custAttr1           -0.000006115          0.000001940  -3.152              0.00162 ** 
# field1               0.029803992          0.040177439   0.742              0.45820    
# field2              -0.342942905          0.084182567  -4.074            0.0000463 ***
# flag1                0.824994619          0.098115654   8.408 < 0.0000000000000002 ***
# field3              -0.000017106          0.000010100  -1.694              0.09032 .  
# field4               0.008074526          0.006499227   1.242              0.21410    
# indicator1           0.358956378          0.109910123   3.266              0.00109 ** 
# indicator2           0.582372231          0.212708841   2.738              0.00618 ** 
# flag2               -1.158063999          0.092551210 -12.513 < 0.0000000000000002 ***
# flag3                0.315826591          0.091009203   3.470              0.00052 ***
# flag4               -0.658021902          0.342822000  -1.919              0.05493 .  
# flag5                0.001857042          0.000428659   4.332            0.0000148 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 6486.8  on 24550  degrees of freedom
# Residual deviance: 6112.4  on 24535  degrees of freedom
# AIC: 6144.4
# 
# Number of Fisher Scoring iterations: 8

#### SVMA RADIAL
ucsd_svm_rad <- train(Class ~ .,
                         data = ucsd_train,
                         method = "svmRadial",
                         preProc = c("center", "scale"),
                         verbose = FALSE,
                         metric = "ROC", 
                         trControl = ctrl_ucsd)
radial_results <- predict(ucsd_svm_rad, newdata = ucsd_test)
conf_matr_radial <- confusionMatrix(radial_results, ucsd_test$Class)
conf_matr_radial
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15869   438
# X2    20    40
# 
# Accuracy : 0.972              
# 95% CI : (0.9694, 0.9745)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.183              
# 
# Kappa : 0.1431             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.99874            
# Specificity : 0.08368            
# Pos Pred Value : 0.97314            
# Neg Pred Value : 0.66667            
# Prevalence : 0.97079            
# Detection Rate : 0.96957            
# Detection Prevalence : 0.99633            
# Balanced Accuracy : 0.54121            
# 
# 'Positive' Class : X1 

radial_results_prob <- predict(ucsd_svm_rad, newdata = ucsd_test, type = "prob")
radial_results_probs <- ifelse(radial_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_radial2 <- confusionMatrix(radial_results_probs, ucsd_test$Class)
conf_matr_radial2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15832   408
# X2    57    70
# 
# Accuracy : 0.9716             
# 95% CI : (0.9689, 0.9741)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.2825             
# 
# Kappa : 0.2219             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9964             
# Specificity : 0.1464             
# Pos Pred Value : 0.9749             
# Neg Pred Value : 0.5512             
# Prevalence : 0.9708             
# Detection Rate : 0.9673             
# Detection Prevalence : 0.9922             
# Balanced Accuracy : 0.5714             
# 
# 'Positive' Class : X1     

## RADIAL WEIGHTS
ucsd_svm_rad_w <- train(Class ~ .,
                      data = ucsd_train,
                      method = "svmRadialWeights",
                      preProc = c("center", "scale"),
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)

radial_results_w <- predict(ucsd_svm_rad_w, newdata = ucsd_test)
conf_matr_radial_w <- confusionMatrix(radial_results_w, ucsd_test$Class)
conf_matr_radial_w
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15888   476
# X2     1     2
# 
# Accuracy : 0.9709             
# 95% CI : (0.9682, 0.9734)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.4936             
# 
# Kappa : 0.008              
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.999937           
# Specificity : 0.004184           
# Pos Pred Value : 0.970912           
# Neg Pred Value : 0.666667           
# Prevalence : 0.970795           
# Detection Rate : 0.970734           
# Detection Prevalence : 0.999817           
# Balanced Accuracy : 0.502061           
#
#'Positive' Class : X1  
radial_results_prob_w <- predict(ucsd_svm_rad, newdata = ucsd_test, type = "prob")
radial_results_probs_w <- ifelse(radial_results_prob_w$X2 > 0.1, "X2", "X1")
conf_matr_radial2_w <- confusionMatrix(radial_results_probs_w, ucsd_test$Class)
conf_matr_radial2_w
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15832   408
# X2    57    70
# 
# Accuracy : 0.9716             
# 95% CI : (0.9689, 0.9741)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.2825             
# 
# Kappa : 0.2219             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9964             
# Specificity : 0.1464             
# Pos Pred Value : 0.9749             
# Neg Pred Value : 0.5512             
# Prevalence : 0.9708             
# Detection Rate : 0.9673             
# Detection Prevalence : 0.9922             
# Balanced Accuracy : 0.5714             
# 
# 'Positive' Class : X1 
ucsd_svm_rad %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.7477

plot(roc(ucsd_test$Class, predict(ucsd_svm_rad, ucsd_test,type = "prob")[,"X2"]))

### RANDOM FOREST
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_randfor <- train(Class ~ .,
                      data = ucsd_train,
                      method = "rf",
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

#Fitting mtry = 8 on full training set
randfor_results <- predict(ucsd_xgboost, newdata = ucsd_test)
conf_matr_randfor <- confusionMatrix(randfor_results, ucsd_test$Class)
conf_matr_randfor
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15863   401
# X2    26    77
# 
# Accuracy : 0.9739               
# 95% CI : (0.9714, 0.9763)     
# No Information Rate : 0.9708               
# P-Value [Acc > NIR] : 0.008685             
# 
# Kappa : 0.2574               
# Mcnemar's Test P-Value : < 0.00000000000000022
# 
# Sensitivity : 0.9984               
# Specificity : 0.1611               
# Pos Pred Value : 0.9753               
# Neg Pred Value : 0.7476               
# Prevalence : 0.9708               
# Detection Rate : 0.9692               
# Detection Prevalence : 0.9937               
# Balanced Accuracy : 0.5797               
# 
# 'Positive' Class : X1       

randfor_results_prob <- predict(ucsd_randfor, newdata = ucsd_test, type = "prob")
randfor_results_probs <- ifelse(randfor_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_randfor2 <- confusionMatrix(randfor_results_probs, ucsd_test$Class)
conf_matr_randfor2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15339   158
# X2   550   320
# 
# Accuracy : 0.9567             
# 95% CI : (0.9535, 0.9598)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.4542             
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.9654             
#             Specificity : 0.6695             
#          Pos Pred Value : 0.9898             
#          Neg Pred Value : 0.3678             
#              Prevalence : 0.9708             
#          Detection Rate : 0.9372             
#    Detection Prevalence : 0.9468             
#       Balanced Accuracy : 0.8174             
#                                              
#        'Positive' Class : X1          

ucsd_randfor %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
# Area under the curve: 0.8952

plot(roc(ucsd_test$Class, predict(ucsd_randfor, ucsd_test,type = "prob")[,"X2"]))


####### NNET
nnet_grid <- expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), .size = c(3, 5, 10, 20))
ucsd_nnet <- train(Class ~ .,
                   data = ucsd_train,
                   method = "nnet",
                   linout = FALSE,
                   verbose = FALSE,
                   tuneGrid = nnet_grid,
                   metric = "ROC", 
                   trControl = ctrl_ucsd)

nnet_results <- predict(ucsd_nnet, newdata = ucsd_test)
conf_matr_nnet <- confusionMatrix(nnet_results, ucsd_test$Class)
conf_matr_nnet
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15889   478
# X2     0     0
# 
# Accuracy : 0.9708             
# 95% CI : (0.9681, 0.9733)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.5122             
# 
# Kappa : 0                  
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 1.0000             
# Specificity : 0.0000             
# Pos Pred Value : 0.9708             
# Neg Pred Value :    NaN             
# Prevalence : 0.9708             
# Detection Rate : 0.9708             
# Detection Prevalence : 1.0000             
# Balanced Accuracy : 0.5000             
# 
# 'Positive' Class : X1    
nnet_results_prob <- predict(ucsd_nnet, newdata = ucsd_test, type = "prob")
nnet_results_probs <- ifelse(nnet_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_nnet2 <- confusionMatrix(nnet_results_probs, ucsd_test$Class)
conf_matr_nnet2


############ GBM
ucsd_gbm <- train(Class ~ .,
                  data = ucsd_train,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC", 
                  trControl = ctrl_ucsd)

gbm_results <- predict(ucsd_gbm, newdata = ucsd_test)
conf_matr_gbm <- confusionMatrix(gbm_results, ucsd_test$Class)
conf_matr_gbm
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15879   444
# X2    10    34
# 
# Accuracy : 0.9723             
# 95% CI : (0.9696, 0.9747)   
# No Information Rate : 0.9708             
# P-Value [Acc > NIR] : 0.1373             
# 
# Kappa : 0.126              
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.99937            
#             Specificity : 0.07113            
#          Pos Pred Value : 0.97280            
#          Neg Pred Value : 0.77273            
#              Prevalence : 0.97079            
#          Detection Rate : 0.97018            
#    Detection Prevalence : 0.99731            
#       Balanced Accuracy : 0.53525            
#                                              
#        'Positive' Class : X1   

gbm_results_prob <- predict(ucsd_gbm, newdata = ucsd_test, type = "prob")
gbm_results_probs <- ifelse(gbm_results_prob$X2 > 0.1, "X2", "X1")
conf_matr_gbm2 <- confusionMatrix(gbm_results_probs, ucsd_test$Class)
conf_matr_gbm2
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15610   329
# X2   279   149
# 
# Accuracy : 0.9629          
# 95% CI : (0.9598, 0.9657)
# No Information Rate : 0.9708          
# P-Value [Acc > NIR] : 1.0000          
# 
# Kappa : 0.3099          
# Mcnemar's Test P-Value : 0.0469          
#                                           
#             Sensitivity : 0.9824          
#             Specificity : 0.3117          
#          Pos Pred Value : 0.9794          
#          Neg Pred Value : 0.3481          
#              Prevalence : 0.9708          
#          Detection Rate : 0.9537          
#    Detection Prevalence : 0.9738          
#       Balanced Accuracy : 0.6471          
#                                           
#        'Positive' Class : X1       
ucsd_gbm %>%
  ucsd_test_roc(data = ucsd_test) %>%
  auc()
#Area under the curve: 0.8042

plot(roc(ucsd_test$Class, predict(ucsd_gbm, ucsd_test,type = "prob")[,"X2"]))
