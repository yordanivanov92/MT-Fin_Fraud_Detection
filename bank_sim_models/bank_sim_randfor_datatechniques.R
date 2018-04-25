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

set.seed(48)
bankSim <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                    header = TRUE,
                    sep = ",")

plyr::count(bankSim, c("category", "fraud"))

plyr::count(bankSim, c("gender", "fraud"))
plyr::count(bankSim, c("merchant", "fraud"))
plyr::count(bankSim, c("age", "fraud"))

bankSim <- bankSim %>%
  select(age, gender, merchant, category, amount, fraud)

bankSim <- bankSim %>%
  filter(category != "'es_transportation'") %>%
  filter(category != "'es_food'") %>%
  filter(category != "'es_contents'") %>%
  filter(gender != "'U'")

### Splitting the data into train and test sets
split = sample.split(bankSim$fraud, SplitRatio = 0.6)

bankSim_train = subset(bankSim, split == TRUE)
bankSim_test = subset(bankSim, split == FALSE)

prop.table(table(bankSim_train$fraud))
prop.table(table(bankSim_test$fraud))


#bankSim_train$customer <- as.factor(bankSim_train$customer)
bankSim_train$age <- as.factor(bankSim_train$age)
bankSim_train$gender <- as.factor(bankSim_train$gender)
bankSim_train$merchant <- as.factor(bankSim_train$merchant)
bankSim_train$category <- as.factor(bankSim_train$category)
bankSim_train$fraud <- ifelse(bankSim_train$fraud == 1, "fraud", "clean")
bankSim_train$fraud <- as.factor(bankSim_train$fraud)


#bankSim_train$customer <- as.factor(bankSim_train$customer)
bankSim_test$age <- as.factor(bankSim_test$age)
bankSim_test$gender <- as.factor(bankSim_test$gender)
bankSim_test$merchant <- as.factor(bankSim_test$merchant)
bankSim_test$category <- as.factor(bankSim_test$category)
bankSim_test$fraud <- ifelse(bankSim_test$fraud == 1, "fraud", "clean")
bankSim_test$fraud <- as.factor(bankSim_test$fraud)


ctrl_bankSim <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 1,
                             allowParallel = TRUE,
                             summaryFunction = twoClassSummary,
                             classProbs = TRUE,
                             verboseIter = TRUE)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_orig_fit <- train(fraud ~ .,
                          data = bankSim_train,
                          method = "rf",
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_bankSim)

stopCluster(cluster)
registerDoSEQ()

randfor_results <- predict(bankSim_orig_fit, newdata = bankSim_test)
confusionMatrix(randfor_results, bankSim_test$fraud)

trellis.par.set(caretTheme())
plot(bankSim_orig_fit, metric = "ROC")

bankSim_imp <- varImp(bankSim_orig_fit, scale = FALSE)
plot(bankSim_imp)

bankSim_test_roc <- function(model, data) {
  roc(data$fraud,
      predict(model, data, type = "prob")[, "fraud"])
}

################## COST SENSITIVE XGBOOST MODEL
bankSim_model_weights <- ifelse(bankSim_train$fraud == "clean",
                                (1/table(bankSim_train$fraud)[1]) * 0.5,
                                (1/table(bankSim_train$fraud)[2]) * 0.5)

ctrl_bankSim$seeds <- bankSim_orig_fit$control$seeds

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_weighted_fit <- train(fraud ~ .,
                              data = bankSim_train,
                              method = "rf",
                              verbose = FALSE,
                              weights = bankSim_model_weights,
                              metric = "ROC",
                              trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()

randfor_results_weighted <- predict(bankSim_weighted_fit, newdata = bankSim_test)
confusionMatrix(randfor_results_weight, bankSim_test$fraud)

trellis.par.set(caretTheme())
plot(bankSim_weighted_fit, metric = "ROC")

bankSim_imp_weighted <- varImp(bankSim_weighted_fit, scale = FALSE)
plot(bankSim_imp_weighted)

############### sampled-down model
ctrl_bankSim$sampling <- "down"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_down_fit <- train(fraud ~ .,
                          data = bankSim_train,
                          method = "rf",
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()

randfor_results_down <- predict(bankSim_down_fit, newdata = bankSim_test)
confusionMatrix(randfor_results_down, bankSim_test$fraud)

trellis.par.set(caretTheme())
plot(bankSim_down_fit, metric = "ROC")

bankSim_imp_down <- varImp(bankSim_down_fit, scale = FALSE)
plot(bankSim_imp_down)

############# sampled-up
ctrl_bankSim$sampling <- "up"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_up_fit <- train(fraud ~ .,
                        data = bankSim_train,
                        method = "rf",
                        verbose = FALSE,
                        metric = "ROC",
                        trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()

randfor_results_up <- predict(bankSim_up_fit, newdata = bankSim_test)
confusionMatrix(randfor_results_up, bankSim_test$fraud)

trellis.par.set(caretTheme())
plot(bankSim_up_fit, metric = "ROC")

bankSim_imp_up <- varImp(bankSim_up_fit, scale = FALSE)
plot(bankSim_imp_up)
############# SMOTE
ctrl_bankSim$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_smote_fit <- train(fraud ~ .,
                           data = bankSim_train,
                           method = "rf",
                           verbose = FALSE,
                           metric = "ROC",
                           trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()

randfor_results_smote <- predict(bankSim_smote_fit, newdata = bankSim_test)
confusionMatrix(randfor_results_smote, bankSim_test$fraud)

trellis.par.set(caretTheme())
plot(bankSim_smote_fit, metric = "ROC")

bankSim_imp_smote <- varImp(bankSim_smote_fit, scale = FALSE)
plot(bankSim_imp_smote)

##########################
bankSim_model_list <- list(original = bankSim_orig_fit,
                           weighted = bankSim_weighted_fit,
                           down = bankSim_down_fit,
                           up = bankSim_up_fit,
                           SMOTE = bankSim_smote_fit)

bankSim_model_list_roc <- bankSim_model_list %>%
  map(bankSim_test_roc, data = bankSim_test)

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


####  Construction the precision/recall graphic
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


