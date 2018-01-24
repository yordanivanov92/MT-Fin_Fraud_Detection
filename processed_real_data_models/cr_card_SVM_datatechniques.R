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
library(e1071)
library(kernlab)
options(scipen=999)

set.seed(48)

credit_card_data <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/dal_pozzlo_real_data_PCA/creditcard.csv",
                             header = TRUE,
                             sep = ",")
credit_card_data <- credit_card_data[sample(nrow(credit_card_data), 50000),]
# Fraud Rate
prop.table(table(credit_card_data$Class))
# Highly imbalanced dataset
# 0           1 
# 0.998272514 0.001727486

# Removing the time step variable
credit_card_data <- credit_card_data %>%
  select(-Time)

split = sample.split(credit_card_data$Class, SplitRatio = 0.6)
cr_card_train <- subset(credit_card_data, split == TRUE)
cr_card_train$Class <- as.factor(cr_card_train$Class)

cr_card_test <- subset(credit_card_data, split == FALSE)
cr_card_test$Class <- as.factor(cr_card_test$Class)

rm(credit_card_data)

feature.names=names(cr_card_train)
for (f in feature.names) {
  if (class(cr_card_train[[f]])=="factor") {
    levels <- unique(c(cr_card_train[[f]]))
    cr_card_train[[f]] <- factor(cr_card_train[[f]],
                                 labels=make.names(levels))
  }
}
feature.names2=names(cr_card_test)
for (f in feature.names2) {
  if (class(cr_card_test[[f]])=="factor") {
    levels <- unique(c(cr_card_test[[f]]))
    cr_card_test[[f]] <- factor(cr_card_test[[f]],
                                labels=make.names(levels))
  }
}

ctrl_cr_card <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 3,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)

cr_card_svm <- train(Class ~ .,
                     data = cr_card_train,
                     method = "svmLinear",
                     preProc = c("center", "scale"),
                     verbose = FALSE,
                     metric = "ROC", 
                     trControl = ctrl_cr_card)

cr_card_svm_rad <- train(Class ~ .,
                         data = cr_card_train,
                         method = "svmRadial",
                         preProc = c("center", "scale"),
                         verbose = FALSE,
                         metric = "ROC", 
                         trControl = ctrl_cr_card)

# Results Original
svm_results <- predict(cr_card_svm, newdata = cr_card_test)
conf_matr_svm <- confusionMatrix(svm_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_svm <- plot(cr_card_svm, metric = "ROC")

svm_imp <- varImp(cr_card_svm, scale = FALSE)
#svm_imp - variable importance is observed
plot(svm_imp)


cr_card_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

auc_svm <- cr_card_svm %>%
  cr_card_test_roc(data = cr_card_test) %>%
  auc()


################## COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
cr_card_model_weights <- ifelse(cr_card_train$Class == "X1",
                                (1/table(cr_card_train$Class)[1]) * 0.5,
                                (1/table(cr_card_train$Class)[2]) * 0.5)

#ctrl_cr_card$seeds <- cr_card_svm$control$seeds


cr_card_svm_weighted_fit <- train(Class ~ .,
                                  data = cr_card_train,
                                  method = "svmLinearWeights",
                                  preProc = c("center", "scale"),
                                  verbose = FALSE,
                                  weights = cr_card_model_weights,
                                  metric = "ROC", 
                                  trControl = ctrl_cr_card)


cr_card_svm_weighted_fit1 <- train(Class ~ .,
                                  data = cr_card_train,
                                  method = "svmLinearWeights",
                                  preProc = c("center", "scale"),
                                  verbose = FALSE,
                                  metric = "ROC", 
                                  trControl = ctrl_cr_card)

# Results CS
svm_weighted_results <- predict(cr_card_svm_weighted_fit, newdata = cr_card_test)
conf_matr_weighted_svm <- confusionMatrix(svm_weighted_results, cr_card_test$Class)

# no difference between the two weighted things
svm_weighted_results1 <- predict(r_card_svm_weighted_fit1, newdata = cr_card_test)
conf_matr_weighted_svm1 <- confusionMatrix(svm_weighted_results1, cr_card_test$Class)

#higher roc values by the second model - without weights in the caret function
trellis.par.set(caretTheme())
train_plot_weighted_svm <- plot(cr_card_svm_weighted_fit, metric = "ROC")

svm_weighted_imp <- varImp(cr_card_svm_weighted_fit, scale = FALSE)
plot(svm_weighted_imp)

auc_svm_weighted <- cr_card_svm_weighted_fit %>%
  cr_card_test_roc(data = cr_card_test) %>%
  auc()

############### sampled-down model
ctrl_cr_card$sampling <- "down"

cr_card_svm_down_fit <- train(Class ~ .,
                              data = cr_card_train,
                              method = "svmLinear",
                              preProc = c("center", "scale"),
                              verbose = FALSE,
                              metric = "ROC",
                              trControl = ctrl_cr_card)


# Results Down
svm_down_results <- predict(cr_card_svm_down_fit, newdata = cr_card_test)
conf_matr_down_svm <- confusionMatrix(svm_down_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_down_svm <- plot(cr_card_svm_down_fit, metric = "ROC")

svm_down_imp <- varImp(cr_card_svm_down_fit, scale = FALSE)
#svm_imp - variable importance is observed
plot(svm_down_imp)

auc_svm_down <- cr_card_svm_down_fit %>%
  cr_card_test_roc(data = cr_card_test) %>%
  auc()

############# sampled-up
ctrl_cr_card$sampling <- "up"

cr_card_svm_up_fit <- train(Class ~ .,
                            data = cr_card_train,
                            method = "svmLinear",
                            preProc = c("center", "scale"),
                            verbose = FALSE,
                            metric = "ROC",
                            trControl = ctrl_cr_card)


# Results Up
svm_up_results <- predict(cr_card_svm_up_fit, newdata = cr_card_test)
conf_matr_up_svm <- confusionMatrix(svm_up_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_up_svm <- plot(cr_card_svm_up_fit, metric = "ROC")

svm_up_imp <- varImp(cr_card_svm_up_fit, scale = FALSE)
#svm_imp - variable importance is observed
plot(svm_up_imp)

auc_svm_up <- cr_card_svm_up_fit %>%
  cr_card_test_roc(data = cr_card_test) %>%
  auc()

############# SMOTE
ctrl_cr_card$sampling <- "smote"

cr_card_svm_smote_fit <- train(Class ~ .,
                               data = cr_card_train,
                               method = "svmLinear",
                               preProc = c("center", "scale"),
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_cr_card)


# Results Up
svm_up_results <- predict(cr_card_svm_up_fit, newdata = cr_card_test)
conf_matr_up_svm <- confusionMatrix(svm_up_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_up_svm <- plot(cr_card_svm_up_fit, metric = "ROC")

svm_up_imp <- varImp(cr_card_svm_up_fit, scale = FALSE)
#svm_imp - variable importance is observed
plot(svm_up_imp)

auc_svm_up <- cr_card_svm_up_fit %>%
  cr_card_test_roc(data = cr_card_test) %>%
  auc()


##############################################################
cr_card_svm_model_list <- list(original = cr_card_svm,
                               weighted = cr_card_svm_weighted_fit,
                               down = cr_card_svm_down_fit,
                               up = cr_card_svm_up_fit,
                               SMOTE = cr_card_svm_smote_fit)
cr_card_svm_model_list_roc <- cr_card_svm_model_list %>%
  map(cr_card_test_roc, data = cr_card_test)

cr_card_svm_model_list_roc %>%
  map(auc)

cr_card_svm_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in cr_card_svm_model_list_roc){
  cr_card_svm_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(cr_card_svm_model_list)[num_mod])
  num_mod <- num_mod + 1
}

cr_card_svm_results_df_roc <- bind_rows(cr_card_svm_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = cr_card_svm_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
cr_card_svm_calc_auprc <- function(model, data) {
  index_class2 <- data$type == "X2"
  index_class1 <- data$type == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$type[index_class2],
           predictions$type[index_class1],
           curve = TRUE)
}

#### ERROR HERE - FIX
cr_card_svm_model_list_pr <- cr_card_svm_model_list %>%
  map(cr_card_svm_calc_auprc, data = cr_card_test)


cr_card_svm_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

cr_card_svm_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in cr_card_svm_model_list_pr) {
  cr_card_svm_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(cr_card_svm_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

cr_card_svm_results_df_pr <- bind_rows(cr_card_svm_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = cr_card_svm_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(cr_card_test$type == "X2")/nrow(cr_card_test),slope = 0, color = "gray", size = 1)


##### HAVE ANOTHER LOOK HERE - NOT ADAPTED
cr_card_svmSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
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
                 method = "svmLinear",
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





