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
                            repeats = 3,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = TRUE)

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

###################### Original model
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost <- train(isFraud ~ .,
                        data = paySim_train,
                        method = "xgbTree",
                        verbose = FALSE,
                        metric = "ROC", 
                        trControl = ctrl_paySim)

stopCluster(cluster)
registerDoSEQ()

paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "X2"])
}

paySim_xgboost %>%
  paySim_test_roc(data = paySim_test) %>%
  auc()

### Original Fit Results
xgboost_results <- predict(paySim_xgboost, newdata = paySim_test)
xgboost_results_conf <- confusionMatrix(xgboost_results, paySim_test$isFraud)

## Optimal ROC Threshold
xgboost_results_prob <- predict(paySim_xgboost, newdata = paySim_test, type = "prob")
xgboost_roc <- paySim_test_roc(paySim_xgboost,data = paySim_test)

ER <- sqrt((1 - xgboost_roc$sensitivities)^2 + (1 - xgboost_roc$specificities)^2)
lowest_dist_corner_pos <- match(min(ER), ER)
opt_thresh <- xgboost_roc$thresholds[lowest_dist_corner_pos]
pred_opt <- ifelse(xgboost_results_prob$X1 >= opt_thresh, "X1", "X2")
xgboost_results_conf_opt <- confusionMatrix(pred_opt, paySim_test$isFraud)
# Gives a lot worse results at first glance - maybe could better with full dataset?

trellis.par.set(caretTheme())
plot(paySim_xgboost, metric = "ROC")

xgboost_imp <- varImp(paySim_xgboost, scale = FALSE)
plot(xgboost_imp)

################## COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
paySim_model_weights <- ifelse(paySim_train$isFraud == "X1",
                                (1/table(paySim_train$isFraud)[1]) * 0.5,
                                (1/table(paySim_train$isFraud)[2]) * 0.5)

ctrl_paySim$seeds <- paySim_xgboost$control$seeds

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost_weighted_fit <- train(isFraud ~ .,
                                     data = paySim_train,
                                     method = "xgbTree",
                                     verbose = FALSE,
                                     weights = paySim_model_weights,
                                     metric = "ROC", 
                                     trControl = ctrl_paySim)
                              
stopCluster(cluster)
registerDoSEQ()

############### sampled-down model
ctrl_paySim$sampling <- "down"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost_down_fit <- train(isFraud ~ .,
                                 data = paySim_train,
                                 method = "xgbTree",
                                 verbose = FALSE,
                                 metric = "ROC",
                                 trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()

############# sampled-up
ctrl_paySim$sampling <- "up"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost_up_fit <- train(isFraud ~ .,
                               data = paySim_train,
                               method = "xgbTree",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()

############# SMOTE
ctrl_paySim$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost_smote_fit <- train(isFraud ~ .,
                                  data = paySim_train,
                                  method = "xgbTree",
                                  verbose = FALSE,
                                  metric = "ROC",
                                  trControl = ctrl_paySim)
stopCluster(cluster)
registerDoSEQ()

paySim_xgboost_model_list <- list(original = paySim_xgboost,
                                  weighted = paySim_xgboost_weighted_fit,
                                  down = paySim_xgboost_down_fit,
                                  up = paySim_xgboost_up_fit,
                                  SMOTE = paySim_xgboost_smote_fit)
paySim_xgboost_model_list_roc <- paySim_xgboost_model_list %>%
  map(paySim_test_roc, data = paySim_test)

paySim_xgboost_model_list_roc %>%
  map(auc)

paySim_xgboost_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in paySim_xgboost_model_list_roc){
  paySim_xgboost_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(paySim_xgboost_model_list)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgboost_results_df_roc <- bind_rows(paySim_xgboost_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_xgboost_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
paySim_xgboost_calc_auprc <- function(model, data) {
  index_class2 <- data$type == "X2"
  index_class1 <- data$type == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$type[index_class2],
           predictions$type[index_class1],
           curve = TRUE)
}

#### ERROR HERE - FIX
paySim_xgboost_model_list_pr <- paySim_xgboost_model_list %>%
  map(paySim_xgboost_calc_auprc, data = paySim_test)


paySim_xgboost_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

paySim_xgboost_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in paySim_xgboost_model_list_pr) {
  paySim_xgboost_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(paySim_xgboost_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgboost_results_df_pr <- bind_rows(paySim_xgboost_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = paySim_xgboost_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(paySim_test$type == "X2")/nrow(paySim_test),slope = 0, color = "gray", size = 1)


##### HAVE ANOTHER LOOK HERE - NOT ADAPTED
paySim_xgboostSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$obs == "X2"
  index_class1 <- data$obs == "X1"
  
  the_curve <- pr.curve(data$type[index_class2],
                        data$type[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}

#Re-initialize control function to remove smote and
# include our new summary function

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = auprcSummary,
                     classProbs = TRUE,
                     seeds = orig_fit$control$seeds)

orig_pr <- train(Class ~ .,
                 data = imbal_train,
                 method = "xgbTree",
                 verbose = FALSE,
                 metric = "AUPRC",
                 trControl = ctrl)

# Get results for auprc on the test set

orig_fit_test <- orig_fit %>%
  calc_auprc(data = imbal_test) %>%
  (function(the_mod) the_mod$auc.integral)

orig_pr_test <- orig_pr %>%
  calc_auprc(data = imbal_test) %>%
  (function(the_mod) the_mod$auc.integral)

# The test errors are the same

identical(orig_fit_test,
          orig_pr_test)
## [1] TRUE
# Because both chose the same
# hyperparameter combination

identical(orig_fit$bestTune,
          orig_pr$bestTune)




################### Results and some graphs


### Weighted fit
xgboost_weight_results <- predict(paySim_xgboost_weighted_fit, newdata = paySim_test)
confusionMatrix(xgboost_weight_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_weight_xgboost, metric = "ROC")

xgboost_weight_imp <- varImp(paySim_xgboost_weighted_fit, scale = FALSE)
plot(xgboost_weight_imp)

### Sampled-down fit
xgboost_down_results <- predict(paySim_xgboost_down_fit, newdata = paySim_test)
confusionMatrix(xgboost_down_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_down_xgboost, metric = "ROC")

xgboost_down_imp <- varImp(paySim_xgboost_down_fit, scale = FALSE)
plot(xgboost_down_imp)

### Sampled-up fit
xgboost_up_results <- predict(paySim_xgboost_up_fit, newdata = paySim_test)
confusionMatrix(xgboost_up_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_up_xgboost, metric = "ROC")

xgboost_up_imp <- varImp(paySim_xgboost_up_fit, scale = FALSE)
plot(xgboost_up_imp)

### Smote fit
xgboost_smote_results <- predict(paySim_xgboost_smote_fit, newdata = paySim_test)
confusionMatrix(xgboost_smote_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_smote_xgboost, metric = "ROC")

xgboost_smote_imp <- varImp(paySim_xgboost_smote_fit, scale = FALSE)
plot(xgboost_smote_imp)