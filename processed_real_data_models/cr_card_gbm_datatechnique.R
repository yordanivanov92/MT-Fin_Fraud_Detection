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
                             repeats = 1,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)

cr_card_gbm <- train(Class ~ .,
                     data = cr_card_train,
                     method = "gbm",
                     verbose = FALSE,
                     metric = "ROC", 
                     trControl = ctrl_cr_card)

# Results Original
gbm_results <- predict(cr_card_gbm, newdata = cr_card_test)
conf_matr_gbm <- confusionMatrix(gbm_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_gbm <- plot(cr_card_gbm, metric = "ROC")
train_plot_gbm

gbm_imp <- varImp(cr_card_gbm, scale = FALSE)
#gbm_imp - variable importance is observed
plot(gbm_imp)

################## COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
cr_card_model_weights <- ifelse(cr_card_train$Class == "X1",
                                (1/table(cr_card_train$Class)[1]) * 0.5,
                                (1/table(cr_card_train$Class)[2]) * 0.5)

ctrl_cr_card$seeds <- cr_card_gbm$control$seeds

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
cr_card_gbm_weighted_fit <- train(Class ~ .,
                                      data = cr_card_train,
                                      method = "gbm",
                                      verbose = FALSE,
                                      weights = cr_card_model_weights,
                                      metric = "ROC", 
                                      trControl = ctrl_cr_card)

stopCluster(cluster)
registerDoSEQ()

# Results CS
gbm_weighted_results <- predict(cr_card_gbm_weighted_fit, newdata = cr_card_test)
conf_matr_weighted_gbm <- confusionMatrix(gbm_weighted_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_weighted_gbm <- plot(cr_card_gbm_weighted_fit, metric = "ROC")
train_plot_weighted_gbm

gbm_weighted_imp <- varImp(cr_card_gbm_weighted_fit, scale = FALSE)
#gbm_imp - variable importance is observed
plot(gbm_weighted_imp)

############### sampled-down model
ctrl_cr_card$sampling <- "down"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
cr_card_gbm_down_fit <- train(Class ~ .,
                                  data = cr_card_train,
                                  method = "gbm",
                                  verbose = FALSE,
                                  metric = "ROC",
                                  trControl = ctrl_cr_card)
stopCluster(cluster)
registerDoSEQ()

# Results Down
gbm_down_results <- predict(cr_card_gbm_down_fit, newdata = cr_card_test)
conf_matr_down_gbm <- confusionMatrix(gbm_down_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_down_gbm <- plot(cr_card_gbm_down_fit, metric = "ROC")
train_plot_down_gbm

gbm_down_imp <- varImp(cr_card_gbm_down_fit, scale = FALSE)
#gbm_imp - variable importance is observed
plot(gbm_down_imp)

############# sampled-up
ctrl_cr_card$sampling <- "up"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
cr_card_gbm_up_fit <- train(Class ~ .,
                                data = cr_card_train,
                                method = "gbm",
                                verbose = FALSE,
                                metric = "ROC",
                                trControl = ctrl_cr_card)
stopCluster(cluster)
registerDoSEQ()

# Results Up
gbm_up_results <- predict(cr_card_gbm_up_fit, newdata = cr_card_test)
conf_matr_up_gbm <- confusionMatrix(gbm_up_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_up_gbm <- plot(cr_card_gbm_up_fit, metric = "ROC")
train_plot_up_gbm

gbm_up_imp <- varImp(cr_card_gbm_up_fit, scale = FALSE)
#gbm_imp - variable importance is observed
plot(gbm_up_imp)

############# SMOTE
ctrl_cr_card$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
cr_card_gbm_smote_fit <- train(Class ~ .,
                                   data = cr_card_train,
                                   method = "gbm",
                                   verbose = FALSE,
                                   metric = "ROC",
                                   trControl = ctrl_cr_card)
stopCluster(cluster)
registerDoSEQ()

# Results Up
gbm_smote_results <- predict(cr_card_gbm_smote_fit, newdata = cr_card_test)
conf_matr_smote_gbm <- confusionMatrix(gbm_smote_results, cr_card_test$Class)

trellis.par.set(caretTheme())
train_plot_smote_gbm <- plot(cr_card_gbm_smote_fit, metric = "ROC")
train_plot_smote_gbm

gbm_smote_imp <- varImp(cr_card_gbm_smote_fit, scale = FALSE)
#gbm_imp - variable importance is observed
plot(gbm_smote_imp)

##############################################################
cr_card_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

cr_card_gbm_model_list <- list(original = cr_card_gbm,
                                   weighted = cr_card_gbm_weighted_fit,
                                   down = cr_card_gbm_down_fit,
                                   up = cr_card_gbm_up_fit,
                                   SMOTE = cr_card_gbm_smote_fit)
cr_card_gbm_model_list_roc <- cr_card_gbm_model_list %>%
  map(cr_card_test_roc, data = cr_card_test)

cr_card_auc_gbm <- as.data.frame(cr_card_gbm_model_list_roc %>% map(auc))
saveRDS(cr_card_auc_gbm, 
        file = paste0(getwd(),"/figures/credit/gbm/cr_card_auc_gbm.rds"))

cr_card_gbm_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in cr_card_gbm_model_list_roc){
  cr_card_gbm_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(cr_card_gbm_model_list)[num_mod])
  num_mod <- num_mod + 1
}

cr_card_gbm_results_df_roc <- bind_rows(cr_card_gbm_results_list_roc)
saveRDS(cr_card_gbm_results_df_roc, 
        file = paste0(getwd(),"/figures/credit/gbm/cr_card_gbm_results_df_roc.rds"))

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = cr_card_gbm_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
cr_card_gbm_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

cr_card_gbm_model_list_pr <- cr_card_gbm_model_list %>%
  map(cr_card_gbm_calc_auprc, data = cr_card_test)


cr_card_PR_gbm <- as.data.frame(cr_card_gbm_model_list_pr %>% map(function(the_mod) the_mod$auc.integral))
saveRDS(cr_card_PR_gbm, 
        file = paste0(getwd(),"/figures/credit/gbm/cr_card_PR_gbm.rds"))

cr_card_gbm_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in cr_card_gbm_model_list_pr) {
  cr_card_gbm_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(cr_card_gbm_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

cr_card_gbm_results_df_pr <- bind_rows(cr_card_gbm_results_list_pr)
saveRDS(cr_card_gbm_results_df_pr, 
        file = paste0(getwd(),"/figures/credit/gbm/cr_card_gbm_results_df_pr.rds"))

ggplot(aes(x = recall, y = precision, group = model), data = cr_card_gbm_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(cr_card_test$Class == "X2")/nrow(cr_card_test),slope = 0, color = "gray", size = 1)

