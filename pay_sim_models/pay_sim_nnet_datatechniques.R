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
paySim <- fread("C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/pay_sim_synthetic/PS_20174392719_1491204439457_log.csv",
                header = TRUE,
                sep = ",")
paySim_small <- paySim[sample(nrow(paySim), 100000), ] 

# Fraud Rate
prop.table(table(paySim_small$isFraud))

# Where does fraud occur -> only in CASH_OUT AND TRANSFER type of payments
plyr::count(paySim_small, c("type", "isFraud"))

fraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 1)), ]
fraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 1)), ]

nofraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 0)), ]
nofraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 0)), ]

frauds <- paySim_small[which(paySim_small$isFraud == 1), ] 
nofrauds <- paySim_small[which(paySim_small$isFraud == 0), ] 



fraud_transfer_dest <- data.frame(paySim[which((paySim$type == "TRANSFER") & (paySim$isFraud == 1)), ]$nameDest)
colnames(fraud_transfer_dest) <- "code"
nofraud_cashout_orig <- data.frame(paySim[which((paySim$type == "CASH_OUT") & (paySim$isFraud == 0)), ]$nameOrig)
colnames(nofraud_cashout_orig) <- "code"
joinned <- merge(fraud_transfer_dest,
                 nofraud_cashout_orig)


# Analysis data
analysis_data_big <- paySim[which(paySim$type == "TRANSFER" | paySim$type == "CASH_OUT"), ]
analysis_data_small <- paySim_small[which(paySim_small$type == "TRANSFER" | paySim_small$type == "CASH_OUT"), ]
rm(paySim)
#drop irrelevant columns
analysis_data_big <- analysis_data_big[, -c("nameOrig", "nameDest", "isFlaggedFraud")]
analysis_data_small <- analysis_data_small[, -c("nameOrig", "nameDest", "isFlaggedFraud")]

analysis_data_big$type <- as.factor(analysis_data_big$type)
analysis_data_small$type <- as.factor(analysis_data_small$type)



analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0)), ]$newbalanceDest <- (-1)
analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0)), ]$oldbalanceDest <- (-1)

analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0)), ]$newbalanceDest <- (-1)
analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0)), ]$oldbalanceDest <- (-1)



analysis_data_big[which(((analysis_data_big$oldbalanceOrg == 0 & analysis_data_big$newbalanceOrg == 0) & analysis_data_big$amount != 0)), ]$newbalanceOrg <- (-1)
analysis_data_big[which(((analysis_data_big$oldbalanceOrg == 0 & analysis_data_big$newbalanceOrg == 0) & analysis_data_big$amount != 0)), ]$oldbalanceOrg <- (-1)

analysis_data_small[which(((analysis_data_small$oldbalanceOrg == 0 & analysis_data_small$newbalanceOrg == 0) & analysis_data_small$amount != 0)), ]$newbalanceOrg <- (-1)
analysis_data_small[which(((analysis_data_small$oldbalanceOrg == 0 & analysis_data_small$newbalanceOrg == 0) & analysis_data_small$amount != 0)), ]$oldbalanceOrg <- (-1)

rm(paySim_small)
rm(fraud_transfer_dest)
rm(nofraud_cashout_orig)
rm(joinned)
rm(frauds)
rm(nofraud_transfer)
rm(nofraud_cashout)
rm(fraud_transfer)
rm(nofrauds)
rm(fraud_cashout)
# Motivated by the possibility of zero-balances serving to differentiate between fraudulent and genuine transactions, 
# we take the data-imputation a step further and create 2 new features (columns) recording errors in 
# the originating and destination accounts for each transaction. 

analysis_data_big$errorBalanceOrig <- analysis_data_big$newbalanceOrig + analysis_data_big$amount - analysis_data_big$oldbalanceOrg
analysis_data_big$errorBalanceDest <- analysis_data_big$oldbalanceDest + analysis_data_big$amount - analysis_data_big$newbalanceDest

analysis_data_small$errorBalanceOrig <- analysis_data_small$newbalanceOrig + analysis_data_small$amount - analysis_data_small$oldbalanceOrg
analysis_data_small$errorBalanceDest <- analysis_data_small$oldbalanceDest + analysis_data_small$amount - analysis_data_small$newbalanceDest

set.seed(434)
split = sample.split(analysis_data_small$isFraud, SplitRatio = 0.6)

paySim_train <- subset(analysis_data_small, split == TRUE)
paySim_train$isFraud <- as.factor(paySim_train$isFraud)
paySim_train$type<-as.factor(paySim_train$type)
paySim_train <- paySim_train[, -c("step")]

paySim_test <- subset(analysis_data_small, split == FALSE)
paySim_test$isFraud <- as.factor(paySim_test$isFraud)
paySim_test$type<-as.factor(paySim_test$type)
paySim_test <- paySim_test[, -c("step")]

ctrl_paySim <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 2,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = TRUE)
nnet_grid <- expand.grid(.decay = c(0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), .size = c(3, 5, 10, 20))

feature.names=names(paySim_train)
for (f in feature.names) {
  if (class(paySim_train[[f]])=="factor") {
    levels <- unique(c(paySim_train[[f]]))
    paySim_train[[f]] <- factor(paySim_train[[f]],
                                labels=make.names(levels))
  }
}
feature.names2=names(paySim_test)
for (f in feature.names2) {
  if (class(paySim_test[[f]])=="factor") {
    levels <- unique(c(paySim_test[[f]]))
    paySim_test[[f]] <- factor(paySim_test[[f]],
                               labels=make.names(levels))
  }
}

rm(analysis_data_big)


paySim_nnet <- train(isFraud ~ .,
                    data = paySim_train,
                    method = "nnet",
                    linout = FALSE,
                    verbose = FALSE,
                    metric = "ROC", 
                    tuneGrid = nnet_grid,
                    trControl = ctrl_paySim)


paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "X2"])
}

paySim_nnet %>%
  paySim_test_roc(data = paySim_test) %>%
  auc()
# Area under the curve: 0.9855
### Original Fit
nnet_results <- predict(paySim_nnet, newdata = paySim_test)
confusionMatrix(nnet_results, paySim_test$isFraud)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 17291    26
# X2     1    24
# 
# Accuracy : 0.9984         
# 95% CI : (0.9977, 0.999)
# No Information Rate : 0.9971         
# P-Value [Acc > NIR] : 0.0002705      
# 
# Kappa : 0.6393         
# Mcnemar's Test P-Value : 0.00000386     
#                                          
#             Sensitivity : 0.9999         
#             Specificity : 0.4800         
#          Pos Pred Value : 0.9985         
#          Neg Pred Value : 0.9600         
#              Prevalence : 0.9971         
#          Detection Rate : 0.9971         
#    Detection Prevalence : 0.9986         
#       Balanced Accuracy : 0.7400         
#                                          
#        'Positive' Class : X1  
trellis.par.set(caretTheme())
plot(paySim_nnet, metric = "ROC")

nnet_imp <- varImp(paySim_nnet, scale = FALSE)
plot(nnet_imp)


################## COST SENSITIVE NNET MODEL
# The penalization costs can be tinkered with
paySim_model_weights <- ifelse(paySim_train$isFraud == "X1",
                               (1/table(paySim_train$isFraud)[1]) * 0.5,
                               (1/table(paySim_train$isFraud)[2]) * 0.5)

ctrl_paySim$seeds <- paySim_nnet$control$seeds

cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
paySim_nnet_weighted_fit <- train(isFraud ~ .,
                                 data = paySim_train,
                                 method = "nnet",
                                 verbose = FALSE,
                                 linout = FALSE,
                                 weights = paySim_model_weights,
                                 metric = "ROC", 
                                 tuneGrid = nnet_grid,
                                 trControl = ctrl_paySim)

stopCluster(cluster)
registerDoSEQ()
### Weighted fit
nnet_weight_results <- predict(paySim_nnet_weighted_fit, newdata = paySim_test)
confusionMatrix(nnet_weight_results, paySim_test$isFraud)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 16923     3
# X2   369    47
# 
# Accuracy : 0.9785             
# 95% CI : (0.9763, 0.9807)   
# No Information Rate : 0.9971             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.1976             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.9787             
# Specificity : 0.9400             
# Pos Pred Value : 0.9998             
# Neg Pred Value : 0.1130             
# Prevalence : 0.9971             
# Detection Rate : 0.9758             
# Detection Prevalence : 0.9760             
# Balanced Accuracy : 0.9593             
# 
# 'Positive' Class : X1              
trellis.par.set(caretTheme())
plot(paySim_nnet_weighted_fit, metric = "ROC")

nnet_weight_imp <- varImp(paySim_nnet_weighted_fit, scale = FALSE)
plot(nnet_weight_imp)

############### sampled-down model
ctrl_paySim$sampling <- "down"
cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
paySim_nnet_down_fit <- train(isFraud ~ .,
                             data = paySim_train,
                             method = "nnet",
                             linout = FALSE,
                             verbose = FALSE,
                             metric = "ROC",
                             tuneGrid = nnet_grid,
                             trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()
### Sampled-down fit
nnet_down_results <- predict(paySim_nnet_down_fit, newdata = paySim_test)
confusionMatrix(nnet_down_results, paySim_test$isFraud)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 15313     0
# X2  1979    50
# 
# Accuracy : 0.8859             
# 95% CI : (0.8811, 0.8906)   
# No Information Rate : 0.9971             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.0427             
# Mcnemar's Test P-Value : <0.0000000000000002
# 
# Sensitivity : 0.88555            
# Specificity : 1.00000            
# Pos Pred Value : 1.00000            
# Neg Pred Value : 0.02464            
# Prevalence : 0.99712            
# Detection Rate : 0.88300            
# Detection Prevalence : 0.88300            
# Balanced Accuracy : 0.94278            
# 
# 'Positive' Class : X1

trellis.par.set(caretTheme())
plot(paySim_nnet_down_fit, metric = "ROC")

nnet_down_imp <- varImp(paySim_nnet_down_fit, scale = FALSE)
plot(nnet_down_imp)

############# sampled-up
ctrl_paySim$sampling <- "up"
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)
paySim_nnet_up_fit <- train(isFraud ~ .,
                           data = paySim_train,
                           method = "nnet",
                           linout = FALSE,
                           verbose = FALSE,
                           metric = "ROC",
                           tuneGrid = nnet_grid,
                           trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()

### Sampled-up fit
nnet_up_results <- predict(paySim_nnet_up_fit, newdata = paySim_test)
confusionMatrix(nnet_up_results, paySim_test$isFraud)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 17239     3
# X2    53    47
# 
# Accuracy : 0.9968          
# 95% CI : (0.9958, 0.9976)
# No Information Rate : 0.9971          
# P-Value [Acc > NIR] : 0.8224          
# 
# Kappa : 0.6252          
# Mcnemar's Test P-Value : 0.00000000005835
#                                           
#             Sensitivity : 0.9969          
#             Specificity : 0.9400          
#          Pos Pred Value : 0.9998          
#          Neg Pred Value : 0.4700          
#              Prevalence : 0.9971          
#          Detection Rate : 0.9941          
#    Detection Prevalence : 0.9942          
#       Balanced Accuracy : 0.9685          
#                                           
#        'Positive' Class : X1             
trellis.par.set(caretTheme())
plot(paySim_nnet_up_fit, metric = "ROC")

nnet_up_imp <- varImp(paySim_nnet_up_fit, scale = FALSE)
plot(nnet_up_imp)

############# SMOTE
ctrl_paySim$sampling <- "smote"
cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
paySim_nnet_smote_fit <- train(isFraud ~ .,
                              data = paySim_train,
                              method = "nnet",
                              linout = FALSE,
                              verbose = FALSE,
                              metric = "ROC",
                              tuneGrid = nnet_grid,
                              trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()

### Smote fit
nnet_smote_results <- predict(paySim_nnet_smote_fit, newdata = paySim_test)
confusionMatrix(nnet_smote_results, paySim_test$isFraud)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    X1    X2
# X1 17192     3
# X2   100    47
# 
# Accuracy : 0.9941             
# 95% CI : (0.9928, 0.9951)   
# No Information Rate : 0.9971             
# P-Value [Acc > NIR] : 1                  
# 
# Kappa : 0.4749             
# Mcnemar's Test P-Value : <0.0000000000000002
#                                              
#             Sensitivity : 0.9942             
#             Specificity : 0.9400             
#          Pos Pred Value : 0.9998             
#          Neg Pred Value : 0.3197             
#              Prevalence : 0.9971             
#          Detection Rate : 0.9914             
#    Detection Prevalence : 0.9915             
#       Balanced Accuracy : 0.9671             
#                                              
#        'Positive' Class : X1              

trellis.par.set(caretTheme())
plot(paySim_nnet_smote_fit, metric = "ROC")

nnet_smote_imp <- varImp(paySim_nnet_smote_fit, scale = FALSE)
plot(nnet_smote_imp)

####################################################

paySim_nnet_model_list <- list(original = paySim_nnet,
                              weighted = paySim_nnet_weighted_fit,
                              down = paySim_nnet_down_fit,
                              up = paySim_nnet_up_fit,
                              SMOTE = paySim_nnet_smote_fit)


paySim_nnet_model_list_roc <- paySim_nnet_model_list %>%
  map(paySim_test_roc, data = paySim_test)

paySim_nnet_model_list_roc %>%
  map(auc)
# $original
# Area under the curve: 0.9855
# 
# $weighted
# Area under the curve: 0.9892
# 
# $down
# Area under the curve: 0.9513
# 
# $up
# Area under the curve: 0.9649
# 
# $SMOTE
# Area under the curve: 0.9962

paySim_nnet_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in paySim_nnet_model_list_roc){
  paySim_nnet_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(paySim_nnet_model_list)[num_mod])
  num_mod <- num_mod + 1
}

paySim_nnet_results_df_roc <- bind_rows(paySim_nnet_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_nnet_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
paySim_nnet_calc_auprc <- function(model, data) {
  index_class2 <- data$isFraud == "X2"
  index_class1 <- data$isFraud == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

paySim_nnet_model_list_pr <- paySim_nnet_model_list %>%
  map(paySim_nnet_calc_auprc, data = paySim_test)

# Precision recall Curve AUC calculation
paySim_nnet_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)
# $original
# [1] 0.7030771
# 
# $weighted
# [1] 0.2299441
# 
# $down
# [1] 0.02719428
# 
# $up
# [1] 0.7307157
# 
# $SMOTE
# [1] 0.7087689


paySim_nnet_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in paySim_nnet_model_list_pr) {
  paySim_nnet_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(paySim_nnet_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

paySim_nnet_results_df_pr <- bind_rows(paySim_nnet_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = paySim_nnet_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(paySim_test$type == "X2")/nrow(paySim_test),slope = 0, color = "gray", size = 1)

#####################################################################################################
paySim_nnetSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$isFraud == "X2"
  index_class1 <- data$isFraud == "X1"
  
  the_curve <- pr.curve(data$X2[index_class2],
                        data$X2[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}

#Re-initialize control function to remove smote and
# include our new summary function

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 2,
                     summaryFunction = paySim_nnetSim_auprcSummary,
                     classProbs = TRUE,
                     seeds = paySim_nnet$control$seeds)

orig_pr <- train(isFraud ~ .,
                 data = paySim_train,
                 method = "nnet",
                 verbose = FALSE,
                 metric = "AUPRC",
                 trControl = ctrl)

# Get results for auprc on the test set

orig_fit_test <- paySim_nnet %>%
  paySim_nnet_calc_auprc(data = paySim_test) %>%
  (function(the_mod) the_mod$auc.integral)

orig_pr_test <- orig_pr %>%
  paySim_nnet_calc_auprc(data = paySim_test) %>%
  (function(the_mod) the_mod$auc.integral)

# The test errors are the same

identical(orig_fit_test,
          orig_pr_test)
## [1] TRUE
# Because both chose the same
# hyperparameter combination

identical(paySim_nnet$bestTune,
          orig_pr$bestTune)




################### Results and some graphs








