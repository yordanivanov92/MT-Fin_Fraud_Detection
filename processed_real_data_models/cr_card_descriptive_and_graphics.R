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

credit_card_data <- fread(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/dal_pozzlo_real_data_PCA/creditcard.csv",
                          header = TRUE,
                          sep = ",")
credit_card_data <- credit_card_data[sample(nrow(credit_card_data), 100000),]

# Fraud Rate
prop.table(table(credit_card_data$Class))
# Highly imbalanced dataset
# 0       1 
# 0.99813 0.00187 

# Removing the time step variable
credit_card_data <- credit_card_data %>%
  select(-Time)

# Amount vs Fraud Boxplots
fraud_vs_amount <- ggplot(credit_card_data, aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot() +
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud")) +
  theme_bw(base_size = 18)
fraud_vs_amount

fraud_vs_low_amount <- ggplot(credit_card_data[credit_card_data$Amount < 2000, ], aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot() + 
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud")) +
  theme_bw(base_size = 24)
fraud_vs_low_amount

fraud_vs_verylow_amount <- ggplot(credit_card_data[credit_card_data$Amount < 100, ], aes(x = Class, y = Amount, group = Class)) +
  geom_boxplot() +
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud")) +
  theme_bw(base_size = 18)
fraud_vs_verylow_amount

# Amount/Fraud Histograms
fraud_low_amount_hist <- ggplot(credit_card_data[credit_card_data$Class == 1 &  credit_card_data$Amount < 1000, ], aes(x = Amount)) +
  geom_histogram(bins = 50) +
  theme_bw(base_size = 18)
fraud_low_amount_hist
nofraud_low_amount_hist <- ggplot(credit_card_data[credit_card_data$Class == 0 & credit_card_data$Amount < 1000, ], aes(x = Amount)) +
  geom_histogram(bins = 50) +
  theme_bw(base_size = 18)
nofraud_low_amount_hist





