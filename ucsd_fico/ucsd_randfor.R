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
library(randomForest)
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

#Getting only those customers that appear more than one
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
                          repeats = 2,
                          summaryFunction = twoClassSummary,
                          #allowParallel = TRUE,
                          classProbs = TRUE,
                          verboseIter = TRUE
)

cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_randfor <- train(Class ~ .,
                      data = ucsd_train,
                      method = "rf",
                      verbose = FALSE,
                      metric = "ROC", 
                      trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

######################################### Randfor PREDICTIONS
randfor_results <- predict(ucsd_randfor, newdata = ucsd_test)
conf_matr_randfor <- confusionMatrix(randfor_results, ucsd_test$Class)
conf_matr_randfor

trellis.par.set(caretTheme())
train_plot_randfor <- plot(ucsd_randfor, metric = "ROC")
train_plot_randfor

randfor_imp <- varImp(ucsd_randfor)
plot(randfor_imp)

############################### COST SENSITIVE RANDFOR MODEL
# The penalization costs can be tinkered with
ucsd_model_weights <- ifelse(ucsd_train$Class == "X1",
                             (1/table(ucsd_train$Class)[1]) * 0.5,
                             (1/table(ucsd_train$Class)[2]) * 0.5)

ctrl_ucsd$seeds <- ucsd_randfor$control$seeds

cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)
ucsd_randfor_weighted_fit <- train(Class ~ .,
                                   data = ucsd_train,
                                   method = "rf",
                                   verbose = FALSE,
                                   weights = ucsd_model_weights,
                                   metric = "ROC", 
                                   trControl = ctrl_ucsd)

stopCluster(cluster)
registerDoSEQ()

randfor_results_weight <- predict(ucsd_randfor_weighted_fit, newdata = ucsd_test)
conf_matr_randfor_weight <- confusionMatrix(randfor_results_weight, ucsd_test$Class)
conf_matr_randfor_weight

trellis.par.set(caretTheme())
train_plot_randfor_weight <- plot(ucsd_randfor_weighted_fit, metric = "ROC")
train_plot_randfor_weight

randfor_imp_weight <- varImp(ucsd_randfor_weighted_fit, scale = FALSE)
plot(randfor_imp_weight)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"
cluster <- makeCluster(detectCores() - 2)
registerDoParallel(cluster)
ucsd_randfor_down_fit <- train(Class ~ .,
                               data = ucsd_train,
                               method = "rf",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

randfor_results_down <- predict(ucsd_randfor_down_fit, newdata = ucsd_test)
conf_matr_randfor_down <- confusionMatrix(randfor_results_down, ucsd_test$Class)
conf_matr_randfor_down

trellis.par.set(caretTheme())
train_plot_randfor_down <- plot(ucsd_randfor_down_fit, metric = "ROC")
train_plot_randfor_down

randfor_imp_down <- varImp(ucsd_randfor_down_fit, scale = FALSE)
plot(randfor_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
cluster <- makeCluster(detectCores() - 2) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_randfor_up_fit <- train(Class ~ .,
                             data = ucsd_train,
                             method = "rf",
                             verbose = FALSE,
                             metric = "ROC",
                             trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

randfor_results_up <- predict(ucsd_randfor_up_fit, newdata = ucsd_test)
conf_matr_randfor_up <- confusionMatrix(randfor_results_up, ucsd_test$Class)
conf_matr_randfor_up

trellis.par.set(caretTheme())
train_plot_randfor_up <- plot(ucsd_randfor_up_fit, metric = "ROC")
train_plot_randfor_up

randfor_imp_up <- varImp(ucsd_randfor_up_fit, scale = FALSE)
plot(randfor_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
ucsd_randfor_smote_fit <- train(Class ~ .,
                                data = ucsd_train,
                                method = "rf",
                                verbose = FALSE,
                                metric = "ROC",
                                trControl = ctrl_ucsd)
stopCluster(cluster)
registerDoSEQ()

randfor_results_smote <- predict(ucsd_randfor_smote_fit, newdata = ucsd_test)
conf_matr_randfor_smote <- confusionMatrix(randfor_results_smote, ucsd_test$Class)
conf_matr_randfor_smote

trellis.par.set(caretTheme())
train_plot_randfor_smote <- plot(ucsd_randfor_smote_fit, metric = "ROC")
train_plot_randfor_smote

randfor_imp_smote <- varImp(ucsd_randfor_smote_fit, scale = FALSE)
plot(randfor_imp_smote)


####################################################################
# ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

ucsd_randfor_model_list <- list(original = ucsd_randfor,
                                weighted = ucsd_randfor_weighted_fit,
                                down = ucsd_randfor_down_fit,
                                up = ucsd_randfor_up_fit,
                                SMOTE = ucsd_randfor_smote_fit)


ucsd_randfor_model_list_roc <- ucsd_randfor_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_auc_randfor <- as.data.frame(ucsd_randfor_model_list_roc %>% map(auc))
saveRDS(ucsd_auc_randfor, file = paste0(getwd(),"/figures/ucsd/randfor/ucsd_auc_randfor.rds"))

ucsd_randfor_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in ucsd_randfor_model_list_roc){
  ucsd_randfor_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(ucsd_randfor_model_list)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_randfor_results_df_roc <- bind_rows(ucsd_randfor_results_list_roc)
saveRDS(ucsd_randfor_results_df_roc, 
        file = paste0(getwd(),"/figures/ucsd/randfor/ucsd_randfor_results_df_roc.rds"))

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_randfor_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
ucsd_randfor_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

ucsd_randfor_model_list_pr <- ucsd_randfor_model_list %>%
  map(ucsd_randfor_calc_auprc, data = ucsd_test)

# Precision recall Curve AUC calculation
ucsd_PR_randfor <- as.data.frame(ucsd_randfor_model_list_pr %>% map(function(the_mod) the_mod$auc.integral))
saveRDS(ucsd_PR_randfor, 
        file = paste0(getwd(),"/figures/ucsd/randfor/ucsd_PR_randfor.rds"))

ucsd_randfor_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in ucsd_randfor_model_list_pr) {
  ucsd_randfor_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(ucsd_randfor_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_randfor_results_df_pr <- bind_rows(ucsd_randfor_results_list_pr)
saveRDS(ucsd_randfor_results_df_pr, 
        file = paste0(getwd(),"/figures/ucsd/randfor/ucsd_randfor_results_df_pr.rds"))

ggplot(aes(x = recall, y = precision, group = model), data = ucsd_randfor_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(ucsd_test$Class == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)
