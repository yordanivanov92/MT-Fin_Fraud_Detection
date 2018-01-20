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
library(ggplot2)
options(scipen=999)

set.seed(48)

credit_card_data <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/dal_pozzlo_real_data_PCA/creditcard.csv",
                             header = TRUE,
                             sep = ",")

# Fraud Rate
prop.table(table(credit_card_data$Class))
# Highly imbalanced dataset
# 0           1 
# 0.998272514 0.001727486

# Removing the time step variable
credit_card_data <- credit_card_data %>%
  select(-Time)

fraud_vs_amount <- ggplot(credit_card_data, aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot()
# slightly strangely, the fraud happens at the lower amounts
 
fraud_vs_low_amount <- ggplot(credit_card_data[credit_card_data$Amount < 300, ], aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot()
fraud_vs_verylow_amount <- ggplot(credit_card_data[credit_card_data$Amount < 100, ], aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot()




