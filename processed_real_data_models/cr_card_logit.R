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
cc_data <- credit_card_data[sample(nrow(bankSim), 100000), ]
# Removing time column
cc_data <- cc_data[, -1]
split = sample.split(cc_data$Class, SplitRatio = 0.6)

cc_data_train = subset(cc_data, split == TRUE)
cc_data_test = subset(cc_data, split == FALSE)

prop.table(table(cc_data_train$Class))
prop.table(table(cc_data_test$Class))

cr_card_logit <- glm(Class ~ ., family = binomial(), data = cc_data_train)

test_results <- predict(cr_card_logit, newdata = cc_data_test)
table(cc_data_test$Class, test_results > 0.5)
