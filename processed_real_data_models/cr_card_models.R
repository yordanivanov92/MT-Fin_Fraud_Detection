# Using real credit card data from Dal Pozzlo
library(dplyr)
library(caret)
library(DMwR) #SMOTE
library(purrr)
library(pROC)
library(gbm)
library(PRROC)
library(caTools)

set.seed(2142)
###########################################################################
############################### BankSim data ##############################
###########################################################################
credit_card_data <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/dal_pozzlo_real_data_PCA/creditcard.csv",
                    header = TRUE,
                    sep = ",")
#cc_data <- credit_card_data[sample(nrow(credit_card_data), 100000), ]
cc_data <- credit_card_data
# Removing time column
cc_data <- cc_data[, -1]
split = sample.split(cc_data$Class, SplitRatio = 0.6)

cc_data_train = subset(cc_data, split == TRUE)
cc_data_test = subset(cc_data, split == FALSE)

prop.table(table(cc_data_train$Class))
prop.table(table(cc_data_test$Class))

ctrl_ccard <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 5,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)


cc_data_train$Class <- ifelse(cc_data_train$Class == 1, "fraud", "clean")
cc_data_train$Class <- as.factor(cc_data_train$Class)

cc_data_test$Class <- ifelse(cc_data_test$Class == 1, "fraud", "clean")
cc_data_test$Class <- as.factor(cc_data_test$Class)

cc_orig <- train(Class ~ .,
                          data = cc_data_train,
                          method = "gbm",
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_ccard)

test_results <- predict(cc_orig, newdata = cc_data_test)
confusionMatrix(test_results, cc_data_test$Class)

cc_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "fraud"])
}

cc_orig %>%
  cc_test_roc(data = cc_data_test) %>%
  auc()


# Handling class imbalance with weighted or sampling methods
cc_data_weights <- ifelse(cc_data_train$Class == "clean",
                                (1/table(cc_data_train$Class)[1]) * 0.5,
                                (1/table(cc_data_train$Class)[2]) * 0.5)

ctrl_ccard$seeds <- cc_orig$control$seeds
#weighted model
cc_weights <- train(Class ~ .,
                              data = cc_data_train,
                              method = "gbm",
                              verbose = FALSE,
                              weights = cc_data_weights,
                              metric = "ROC",
                              trControl = ctrl_ccard)

#sampled-down model
ctrl_ccard$sampling <- "down"
cc_down <- train(Class ~ .,
                          data = cc_data_train,
                          method = "gbm",
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_ccard)

#sampled-up
ctrl_ccard$sampling <- "up"
cc_up <- train(Class ~ .,
                        data = cc_data_train,
                        method = "gbm",
                        verbose = FALSE,
                        metric = "ROC",
                        trControl = ctrl_ccard)

#SMOTE
ctrl_ccard$sampling <- "smote"
cc_smote <- train(Class ~ .,
                           data = cc_data_train,
                           method = "gbm",
                           verbose = FALSE,
                           metric = "ROC",
                           trControl = ctrl_ccard)

cc_model_list <- list(original = cc_orig,
                      weighted = cc_weights,
                      down = cc_down,
                      up = cc_up,
                      SMOTE = cc_smote)
cc_model_list_roc <- cc_model_list %>%
  map(cc_test_roc, data = cc_data_train)

cc_model_list_roc %>%
  map(auc)

cc_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in cc_model_list_roc){
  cc_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(cc_model_list)[num_mod])
  num_mod <- num_mod + 1
}

cc_model_list_roc_df <- bind_rows(cc_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = cc_model_list_roc_df) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


##### the test_results_model do not give probabilities, as the type = "prob" is omitted
####  the predict() gives us directly predictions at a cutoff at 0.5
####  a thing to try is to create confusion matrices at different cutoffs
test_results_orig <- predict(cc_orig, newdata = cc_data_test)
confusionMatrix(test_results_orig, cc_data_test$Class)

test_results_weight <- predict(cc_weights, newdata = cc_data_test)
confusionMatrix(test_results_weight, cc_data_test$Class)

test_results_up <- predict(cc_up, newdata = cc_data_test)
confusionMatrix(test_results_up, cc_data_test$Class)

test_results_down <- predict(cc_down, newdata = cc_data_test)
confusionMatrix(test_results_down, cc_data_test$Class)

test_results_smote <- predict(cc_smote, newdata = cc_data_test)
confusionMatrix(test_results_smote, cc_data_test$Class)



#### second part - more detailed metrics
cc_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "fraud"
  index_class1 <- data$Class == "clean"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$fraud[index_class2],
           predictions$fraud[index_class1],
           curve = TRUE)
}

cc_model_list_pr <- cc_model_list %>%
  map(cc_calc_auprc, data = cc_data_test)


cc_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

cc_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in cc_model_list_pr) {
  cc_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(cc_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

cc_results_df_pr <- bind_rows(cc_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = cc_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(cc_data_test$Class == "fraud")/nrow(cc_data_test),slope = 0, color = "gray", size = 1)


cc_auprcSummary <- function(data, lev = NULL, model = NULL){
  
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
                 data = cc_data_train,
                 method = "gbm",
                 verbose = FALSE,
                 metric = "AUPRC",
                 trControl = ctrl_ccard)

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





