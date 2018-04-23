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

# Getting only those customers that appear more than one
ucsd_data <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1) %>%
  dplyr::inner_join(ucsd_data, by = "custAttr1") %>%
  dplyr::select(-freq)

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
                          repeats = 1,
                          summaryFunction = twoClassSummary,
                          #allowParallel = TRUE,
                          classProbs = TRUE,
                          verboseIter = TRUE
)

nnet_grid <- expand.grid(.decay = c(0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7), 
                         .size = c(3, 5, 10, 20))

ucsd_nnet <- caret::train(Class ~ .,
                          data = ucsd_train,
                          method = "nnet",
                          linout = FALSE,
                          maxit = 2000,
                          verbose = FALSE,
                          metric = "ROC",
                          tuneGrid = nnet_grid,
                          trControl=ctrl_ucsd)

ucsd_train <- ucsd_train %>%
  select(-c(custAttr1, zip1))

# These two features are removed, as if we use them in the analysis, 
# the ANN performs really poor on the test set, predicting only 
# legitimate transactions and ignoring all frauds. 

ucsd_nnet_sub <- caret::train(Class ~ .,
                              data = ucsd_train,
                              method = "nnet",
                              linout = FALSE,
                              maxit = 2000,
                              verbose = FALSE,
                              metric = "ROC",
                              tuneGrid = nnet_grid,
                              trControl=ctrl_ucsd)

############ Neural Network PREDICTIONS ############ 
nnet_results <- predict(ucsd_nnet_sub, newdata = ucsd_test)
conf_matr_nnet <- confusionMatrix(nnet_results, ucsd_test$Class)
conf_matr_nnet

#####################    

trellis.par.set(caretTheme())
train_plot_nnet <- plot(ucsd_nnet_sub, metric = "ROC")
train_plot_nnet

nnet_imp <- varImp(ucsd_nnet)
plot(nnet_imp)

# XGBOOST ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

############################### COST SENSITIVE RANDFOR MODEL
# The penalization costs can be tinkered with
ucsd_model_weights <- ifelse(ucsd_train$Class == "X1",
                             (1/table(ucsd_train$Class)[1]) * 0.5,
                             (1/table(ucsd_train$Class)[2]) * 0.5)

ctrl_ucsd$seeds <- ucsd_nnet_sub$control$seeds

ucsd_nnet_weighted_fit <- train(Class ~ .,
                                data = ucsd_train,
                                method = "nnet",
                                verbose = FALSE,
                                maxit = 2000,
                                weights = ucsd_model_weights,
                                metric = "ROC", 
                                tuneGrid = nnet_grid,
                                trControl = ctrl_ucsd)

nnet_results_weight <- predict(ucsd_nnet_weighted_fit, newdata = ucsd_test)
conf_matr_nnet_weight <- confusionMatrix(nnet_results_weight, ucsd_test$Class)
conf_matr_nnet_weight

trellis.par.set(caretTheme())
train_plot_nnet_weight <- plot(ucsd_nnet_weighted_fit, metric = "ROC")
train_plot_nnet_weight

nnet_imp_weight <- varImp(ucsd_nnet_weighted_fit, scale = FALSE)
plot(nnet_imp_weight)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"
ucsd_nnet_down_fit <- train(Class ~ .,
                            data = ucsd_train,
                            method = "nnet",
                            verbose = FALSE,
                            metric = "ROC",
                            maxit = 2000,
                            tuneGrid = nnet_grid,
                            trControl = ctrl_ucsd)


nnet_results_down <- predict(ucsd_nnet_down_fit, newdata = ucsd_test)
conf_matr_nnet_down <- confusionMatrix(nnet_results_down, ucsd_test$Class)
conf_matr_nnet_down
    

trellis.par.set(caretTheme())
train_plot_nnet_down <- plot(ucsd_nnet_down_fit, metric = "ROC")
train_plot_nnet_down

nnet_imp_down <- varImp(ucsd_nnet_down_fit, scale = FALSE)
plot(nnet_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
ucsd_nnet_up_fit <- train(Class ~ .,
                          data = ucsd_train,
                          method = "nnet",
                          verbose = FALSE,
                          metric = "ROC",
                          maxit = 2000,
                          tuneGrid = nnet_grid,
                          trControl = ctrl_ucsd)

nnet_results_up <- predict(ucsd_nnet_up_fit, newdata = ucsd_test)
conf_matr_nnet_up <- confusionMatrix(nnet_results_up, ucsd_test$Class)
conf_matr_nnet_up

trellis.par.set(caretTheme())
train_plot_nnet_up <- plot(ucsd_nnet_up_fit, metric = "ROC")
train_plot_nnet_up

nnet_imp_up <- varImp(ucsd_nnet_up_fit, scale = FALSE)
plot(nnet_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"

ucsd_nnet_smote_fit <- train(Class ~ .,
                             data = ucsd_train,
                             method = "nnet",
                             verbose = FALSE,
                             metric = "ROC",
                             maxit = 2000,
                             tuneGrid = nnet_grid,
                             trControl = ctrl_ucsd)


nnet_results_smote <- predict(ucsd_nnet_smote_fit, newdata = ucsd_test)
conf_matr_nnet_smote <- confusionMatrix(nnet_results_smote, ucsd_test$Class)
conf_matr_nnet_smote

trellis.par.set(caretTheme())
train_plot_nnet_smote <- plot(ucsd_nnet_smote_fit, metric = "ROC")
train_plot_nnet_smote

nnet_imp_smote <- varImp(ucsd_nnet_smote_fit, scale = FALSE)
plot(nnet_imp_smote)


####################################################################

ucsd_nnet_model_list <- list(original = ucsd_nnet_sub,
                             weighted = ucsd_nnet_weighted_fit,
                             down = ucsd_nnet_down_fit,
                             up = ucsd_nnet_up_fit,
                             SMOTE = ucsd_nnet_smote_fit)


ucsd_nnet_model_list_roc <- ucsd_nnet_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_nnet_model_list_roc %>%
  map(auc)


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
  geom_abline(intercept = sum(ucsd_test$Class == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)
