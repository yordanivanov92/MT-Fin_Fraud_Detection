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

ctrl_paySim <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 1,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = TRUE)


paySim_xgb <- train(isFraud ~ .,
                    data = paySim_train,
                    method = "xgbTree",
                    verbose = FALSE,
                    metric = "ROC", 
                    trControl = ctrl_paySim)


# Area under the curve: 1
### Original Fit
xgb_results <- predict(paySim_xgb, newdata = paySim_test)
confusionMatrix(xgb_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgb, metric = "ROC")

xgb_imp <- varImp(paySim_xgb, scale = FALSE)
plot(xgb_imp)


paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "X2"])
}
################## COST SENSITIVE XGB MODEL
# The penalization costs can be tinkered with
paySim_model_weights <- ifelse(paySim_train$isFraud == "X1",
                               (1/table(paySim_train$isFraud)[1]) * 0.5,
                               (1/table(paySim_train$isFraud)[2]) * 0.5)

ctrl_paySim$seeds <- paySim_xgb$control$seeds

paySim_xgb_weighted_fit <- train(isFraud ~ .,
                                 data = paySim_train,
                                 method = "xgbTree",
                                 verbose = FALSE,
                                 weights = paySim_model_weights,
                                 metric = "ROC", 
                                 trControl = ctrl_paySim)

### Weighted fit
xgb_weight_results <- predict(paySim_xgb_weighted_fit, newdata = paySim_test)
confusionMatrix(xgb_weight_results, paySim_test$isFraud)
            
trellis.par.set(caretTheme())
plot(paySim_xgb_weighted_fit, metric = "ROC")

xgb_weight_imp <- varImp(paySim_xgb_weighted_fit, scale = FALSE)
plot(xgb_weight_imp)

############### sampled-down model
ctrl_paySim$sampling <- "down"

paySim_xgb_down_fit <- train(isFraud ~ .,
                             data = paySim_train,
                             method = "xgbTree",
                             verbose = FALSE,
                             metric = "ROC",
                             trControl = ctrl_paySim)

### Sampled-down fit
xgb_down_results <- predict(paySim_xgb_down_fit, newdata = paySim_test)
confusionMatrix(xgb_down_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgb_down_fit, metric = "ROC")

xgb_down_imp <- varImp(paySim_xgb_down_fit, scale = FALSE)
plot(xgb_down_imp)

############# sampled-up
ctrl_paySim$sampling <- "up"

paySim_xgb_up_fit <- train(isFraud ~ .,
                           data = paySim_train,
                           method = "xgbTree",
                           verbose = FALSE,
                           metric = "ROC",
                           trControl = ctrl_paySim)


### Sampled-up fit
xgb_up_results <- predict(paySim_xgb_up_fit, newdata = paySim_test)
confusionMatrix(xgb_up_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgb_up_fit, metric = "ROC")

xgb_up_imp <- varImp(paySim_xgb_up_fit, scale = FALSE)
plot(xgb_up_imp)

############# SMOTE
ctrl_paySim$sampling <- "smote"

paySim_xgb_smote_fit <- train(isFraud ~ .,
                              data = paySim_train,
                              method = "xgbTree",
                              verbose = FALSE,
                              metric = "ROC",
                              trControl = ctrl_paySim)


### Smote fit
xgb_smote_results <- predict(paySim_xgb_smote_fit, newdata = paySim_test)
confusionMatrix(xgb_smote_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgb_smote_fit, metric = "ROC")

xgb_smote_imp <- varImp(paySim_xgb_smote_fit, scale = FALSE)
plot(xgb_smote_imp)

####################################################

paySim_xgb_model_list <- list(original = paySim_xgb,
                              weighted = paySim_xgb_weighted_fit,
                              down = paySim_xgb_down_fit,
                              up = paySim_xgb_up_fit,
                              SMOTE = paySim_xgb_smote_fit)


paySim_xgb_model_list_roc <- paySim_xgb_model_list %>%
  map(paySim_test_roc, data = paySim_test)

paySim_xgb_model_list_roc %>%
  map(auc)

paySim_xgb_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in paySim_xgb_model_list_roc){
  paySim_xgb_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(paySim_xgb_model_list)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgb_results_df_roc <- bind_rows(paySim_xgb_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_xgb_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
paySim_xgb_calc_auprc <- function(model, data) {
  index_class2 <- data$isFraud == "X2"
  index_class1 <- data$isFraud == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

paySim_xgb_model_list_pr <- paySim_xgb_model_list %>%
  map(paySim_xgb_calc_auprc, data = paySim_test)

# Precision recall Curve AUC calculation
paySim_xgb_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

paySim_xgb_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in paySim_xgb_model_list_pr) {
  paySim_xgb_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(paySim_xgb_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgb_results_df_pr <- bind_rows(paySim_xgb_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = paySim_xgb_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(paySim_test$type == "X2")/nrow(paySim_test),slope = 0, color = "gray", size = 1)

