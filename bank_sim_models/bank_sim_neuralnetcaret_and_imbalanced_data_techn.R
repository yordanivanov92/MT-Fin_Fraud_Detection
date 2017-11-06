library(dplyr)
library(caret)
library(DMwR) #SMOTE
library(purrr)
library(pROC)
library(gbm)
library(PRROC)
library(caTools)
library(nnet)

set.seed(3)
###########################################################################
############################### BankSim data ##############################
###########################################################################
bankSim <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                    header = TRUE,
                    sep = ",")
bankSim_small <- bankSim[sample(nrow(bankSim), 10000), ]

# Keep only relevant columns
bankSim_model <- bankSim_small[, 2:10]
bankSim_model <- bankSim_model[, c("age", "gender", "merchant", "category", "amount", "fraud")]
split = sample.split(bankSim_model$fraud, SplitRatio = 0.6)

bankSim_train = subset(bankSim_model, split == TRUE)
bankSim_test = subset(bankSim_model, split == FALSE)

prop.table(table(bankSim_train$fraud))

ctrl_bankSim <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 5,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)

bankSim_train$fraud <- as.factor(bankSim_train$fraud)
#bankSim_train$customer <- as.factor(bankSim_train$customer)
bankSim_train$age <- as.factor(bankSim_train$age)
bankSim_train$gender <- as.factor(bankSim_train$gender)
bankSim_train$merchant <- as.factor(bankSim_train$merchant)
bankSim_train$category <- as.factor(bankSim_train$category)
bankSim_train$fraud <- ifelse(bankSim_train$fraud == 1, "fraud", "clean")

bankSim_test$fraud <- as.factor(bankSim_test$fraud)
#bankSim_train$customer <- as.factor(bankSim_train$customer)
bankSim_test$age <- as.factor(bankSim_test$age)
bankSim_test$gender <- as.factor(bankSim_test$gender)
bankSim_test$merchant <- as.factor(bankSim_test$merchant)
bankSim_test$category <- as.factor(bankSim_test$category)
bankSim_test$fraud <- ifelse(bankSim_test$fraud == 1, "fraud", "clean")

bankSim_orig_fit <- train(fraud ~ .,
                          data = bankSim_train,
                          method = "nnet",
                          linout = FALSE,
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_bankSim)
test_results <- predict(bankSim_orig_fit, newdata = bankSim_test)
confusionMatrix(test_results, bankSim_test$fraud)

bankSim_test_roc <- function(model, data) {
  roc(data$fraud,
      predict(model, data, type = "prob")[, "fraud"])
}

bankSim_orig_fit %>%
  bankSim_test_roc(data = bankSim_test) %>%
  auc()


# Handling class imbalance with weighted or sampling methods
bankSim_model_weights <- ifelse(bankSim_train$fraud == "clean",
                                (1/table(bankSim_train$fraud)[1]) * 0.5,
                                (1/table(bankSim_train$fraud)[2]) * 0.5)

ctrl_bankSim$seeds <- bankSim_orig_fit$control$seeds
#weighted model
bankSim_weighted_fit <- train(fraud ~ .,
                              data = bankSim_train,
                              method = "nnet",
                              linout = FALSE,
                              verbose = FALSE,
                              weights = bankSim_model_weights,
                              metric = "ROC",
                              trControl = ctrl_bankSim)

#sampled-down model
ctrl_bankSim$sampling <- "down"
bankSim_down_fit <- train(fraud ~ .,
                          data = bankSim_train,
                          method = "nnet",
                          linout = FALSE,
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_bankSim)

#sampled-up
ctrl_bankSim$sampling <- "up"
bankSim_up_fit <- train(fraud ~ .,
                        data = bankSim_train,
                        method = "nnet",
                        linout = FALSE,
                        verbose = FALSE,
                        metric = "ROC",
                        trControl = ctrl_bankSim)

#SMOTE
ctrl_bankSim$sampling <- "smote"
bankSim_smote_fit <- train(fraud ~ .,
                           data = bankSim_train,
                           method = "nnet",
                           linout = FALSE,
                           verbose = FALSE,
                           metric = "ROC",
                           trControl = ctrl_bankSim)

bankSim_model_list <- list(original = bankSim_orig_fit,
                           weighted = bankSim_weighted_fit,
                           down = bankSim_down_fit,
                           up = bankSim_up_fit,
                           SMOTE = bankSim_smote_fit)
bankSim_model_list_roc <- bankSim_model_list %>%
  map(bankSim_test_roc, data = bankSim_train)

bankSim_model_list_roc %>%
  map(auc)

bankSim_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in bankSim_model_list_roc){
  bankSim_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(bankSim_model_list)[num_mod])
  num_mod <- num_mod + 1
}

bankSim_results_df_roc <- bind_rows(bankSim_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = bankSim_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


#### second part - more detailed metrics
bankSim_calc_auprc <- function(model, data) {
  index_class2 <- data$fraud == "fraud"
  index_class1 <- data$fraud == "clean"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$fraud[index_class2],
           predictions$fraud[index_class1],
           curve = TRUE)
}

bankSim_model_list_pr <- bankSim_model_list %>%
  map(bankSim_calc_auprc, data = bankSim_test)


bankSim_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

bankSim_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in bankSim_model_list_pr) {
  bankSim_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(bankSim_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

bankSim_results_df_pr <- bind_rows(bankSim_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = bankSim_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(bankSim_test$fraud == "fraud")/nrow(bankSim_test),slope = 0, color = "gray", size = 1)


bankSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$obs == "fraud"
  index_class1 <- data$obs == "clean"
  
  the_curve <- pr.curve(data$fraud[index_class2],
                        data$fraud[index_class1],
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
                 method = "gbm",
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




test_results_orig <- predict(bankSim_orig_fit, newdata = bankSim_test)
confusionMatrix(test_results_orig, bankSim_test$fraud)

test_results_weight <- predict(bankSim_weighted_fit, newdata = bankSim_test)
confusionMatrix(test_results_weight, bankSim_test$fraud)

test_results_down <- predict(bankSim_down_fit, newdata = bankSim_test)
confusionMatrix(test_results_down, bankSim_test$fraud)

test_results_up <- predict(bankSim_up_fit, newdata = bankSim_test)
confusionMatrix(test_results_up, bankSim_test$fraud)

test_results_smote <- predict(bankSim_smote_fit, newdata = bankSim_test)
confusionMatrix(test_results_up, bankSim_test$fraud)
