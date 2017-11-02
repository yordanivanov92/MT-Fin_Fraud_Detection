library(dplyr)
library(caret)
library(DMwR) #SMOTE
library(purrr)
library(pROC)
library(gbm)
library(PRROC)
library(caTools)
library(neuralnet)

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
