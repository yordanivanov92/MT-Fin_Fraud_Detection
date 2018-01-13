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

set.seed(1)
###########################################################################
############################### BankSim data ##############################
###########################################################################
bankSim <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                    header = TRUE,
                    sep = ",")

plyr::count(bankSim, c("category", "fraud"))
# es_contents - no fraud
# es_food - no fraud
# es_transportation - no fraud


plyr::count(bankSim, c("gender", "fraud"))
# the U gender - no frauds commited
plyr::count(bankSim, c("merchant", "fraud"))
# some merchants - no fraud
plyr::count(bankSim, c("age", "fraud"))
# each age category as commited fraud

##### NEEDS BIGGER COMPUTING POWER IF THE CUSTOMER CATEGORY IS TO BE INCLUDED
# bankSim <- bankSim %>%
#   select(customer, age, gender, merchant, category, amount, fraud)

bankSim <- bankSim %>%
  select(age, gender, merchant, category, amount, fraud)

bankSim <- bankSim %>%
  filter(category != "'es_transportation'") %>%
  filter(category != "'es_food'") %>%
  filter(category != "'es_contents'") %>%
  filter(gender != "'U'")


###################################################################
filtered_customers <- plyr::count(bankSim, c("customer", "fraud"))
dupl_data_customers <- filtered_customers[duplicated(filtered_customers[, "customer"]), ]

#getting those customers that have exhibited fraud
bankSim_filter_customers <- bankSim %>%
  filter(customer %in% dupl_data_customers[, "customer"])  %>%
  select(customer) %>%
  distinct(customer) %>%
  arrange(customer)

bankSim <- bankSim %>%
  filter(customer %in% bankSim_filter_customers[, "customer"])
rm(filtered_customers)
rm(dupl_data_customers)
rm(bankSim_filter_customers)
###################
filtered_merchant <- plyr::count(bankSim, c("merchant", "fraud"))
dupl_data_merchant <- filtered_merchant[duplicated(filtered_merchant[, "merchant"]), ]

#getting those merchants that have exhibited fraud
bankSim_filter_merchant <- bankSim %>%
  filter(merchant %in% dupl_data_merchant[, "merchant"])  %>%
  select(merchant) %>%
  distinct(merchant) %>%
  arrange(merchant)

bankSim <- bankSim %>%
  filter(merchant %in% bankSim_filter_merchant[, "merchant"])

rm(filtered_merchant)
rm(dupl_data_merchant)
rm(bankSim_filter_merchant)
#############################

# Checking fraud by amount
ggplot(bankSim, aes(x = fraud, y = amount, group = fraud)) +
  geom_boxplot()
# Fraud looks like it has a higher average amount and a lot more and larger outliers

ggplot(bankSim[bankSim$amount < 2000,], aes(x = fraud, y = amount, group = fraud)) +
  geom_boxplot()
# Fraud is associated with larger amounts

ggplot(bankSim[bankSim$amount < 2000, ], aes(x = as.factor(fraud), y = amount, group = fraud)) +
  geom_boxplot(aes(group = interaction(category, fraud), fill = category))


#######################
summary_1 <- bankSim %>%
  group_by(amount) %>%
  dplyr::summarize(Total = n())

ggplot(summary_1, aes(amount, Total)) +
  geom_bar(stat = "identity", width = 1)
