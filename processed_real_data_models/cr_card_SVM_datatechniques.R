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
                             repeats = 1,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)

### Models Start
cr_card_svm <- train(Class ~ .,
                     data = cr_card_train,
                     method = "svmLinear",
                     preProc = c("center", "scale"),
                     verbose = FALSE,
                     metric = "ROC", 
                     trControl = ctrl_cr_card)

# Results Original
svm_results <- predict(cr_card_svm, newdata = cr_card_test)
conf_matr_svm <- confusionMatrix(svm_results, cr_card_test$Class)

svm_imp <- varImp(cr_card_svm, scale = FALSE)
plot(svm_imp)

cr_card_svm_rad <- train(Class ~ .,
                         data = cr_card_train,
                         method = "svmRadial",
                         preProc = c("center", "scale"),
                         verbose = FALSE,
                         metric = "ROC", 
                         trControl = ctrl_cr_card)

svm_results_rad <- predict(cr_card_svm_rad, newdata = cr_card_test)
conf_matr_svm_rad <- confusionMatrix(svm_results_rad, cr_card_test$Class) # radial kernel performing better


trellis.par.set(caretTheme())
train_plot_svm_rad <- plot(cr_card_svm_rad, metric = "ROC")
train_plot_svm_rad

svm_imp_rad <- varImp(cr_card_svm_rad, scale = FALSE)
plot(svm_imp_rad)



################## COST SENSITIVE SVM MODEL
ctrl_cr_card$seeds <- cr_card_svm_weighted_fit$control$seeds
cr_card_svm_weighted_fit <- train(Class ~ .,
                                  data = cr_card_train,
                                  method = "svmLinearWeights",
                                  preProc = c("center", "scale"),
                                  verbose = FALSE,
                                  metric = "ROC", 
                                  trControl = ctrl_cr_card)


svm_weighted_results <- predict(cr_card_svm_weighted_fit, newdata = cr_card_test)
conf_matr_weighted_svm <- confusionMatrix(svm_weighted_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_weighted_svm <- plot(cr_card_svm_weighted_fit, metric = "ROC")
train_plot_weighted_svm

svm_weighted_imp <- varImp(cr_card_svm_weighted_fit, scale = FALSE)
plot(svm_weighted_imp)


cr_card_svm_weighted_rad_fit <- train(Class ~ .,
                                  data = cr_card_train,
                                  method = "svmRadialWeights",
                                  preProc = c("center", "scale"),
                                  verbose = FALSE,
                                  metric = "ROC", 
                                  trControl = ctrl_cr_card)

svm_weighted_results_rad <- predict(cr_card_svm_weighted_rad_fit, newdata = cr_card_test)
conf_matr_weighted_svm_rad <- confusionMatrix(svm_weighted_results_rad, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_weighted_svm_rad <- plot(cr_card_svm_weighted_rad_fit, metric = "ROC")
train_plot_weighted_svm_rad

svm_weighted_imp_rad <- varImp(cr_card_svm_weighted_rad_fit, scale = FALSE)
plot(svm_weighted_imp_rad)

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

svm_down_imp <- varImp(cr_card_svm_down_fit, scale = FALSE)
plot(svm_down_imp)

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

svm_up_imp <- varImp(cr_card_svm_up_fit, scale = FALSE)
plot(svm_up_imp)

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

svm_up_imp <- varImp(cr_card_svm_up_fit, scale = FALSE)
plot(svm_up_imp)

##############################################################
cr_card_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

cr_card_svm_model_list <- list(original = cr_card_svm,
                               weighted = cr_card_svm_weighted_fit,
                               original_rad = cr_card_svm_rad,
                               weighted_rad = cr_card_svm_weighted_rad_fit,
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
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

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
