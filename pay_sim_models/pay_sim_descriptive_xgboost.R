# Using PaySim from Edgar Lopez
library(data.table)
library(dplyr)
library(caret)
library(DMwR) #SMOTE
library(purrr)
library(pROC)
library(gbm)
library(PRROC)
library(caTools)
library(doParallel)
library(parallel)
library(plyr)
library(GGally)
library(plotly)
library(reshape2)
library(xgboost)
options(scipen=999)

set.seed(45)
paySim <- fread("C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/pay_sim_synthetic/PS_20174392719_1491204439457_log.csv",
                    header = TRUE,
                    sep = ",")
paySim_small <- paySim[sample(nrow(paySim), 100000), ] 
#rm(paySim)

# Fraud Rate
prop.table(table(paySim_small$isFraud))

# Where does fraud occur -> only in CASH_OUT AND TRANSFER type of payments
plyr::count(paySim_small, c("type", "isFraud"))

# Checking fraud by amount
ggplot(paySim_small, aes(x = isFraud, y = amount, group = isFraud)) +
  geom_boxplot()

ggplot(paySim_small[paySim_small$amount < 2000000, ], aes(x = isFraud, y = amount, group = isFraud, fill = isFraud)) +
  geom_boxplot()

ggplot(paySim_small[paySim_small$amount < 2000000, ], aes(x = as.factor(isFraud), y = amount, group = isFraud)) +
  geom_boxplot(aes(group = interaction(type, isFraud), fill = type))

paySim_small1 <- paySim_small[paySim_small$amount < 2000000, ]
paySim_small1$isFraud <- as.factor(paySim_small1$isFraud)
ggpairs(paySim_small1, columns = c("amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"), mapping = aes(color = isFraud))


fraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 1)), ]
fraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 1)), ]

nofraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 0)), ]
nofraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 0)), ]

frauds <- paySim_small[which(paySim_small$isFraud == 1), ] 
nofrauds <- paySim_small[which(paySim_small$isFraud == 0), ] 
# It was stated in the paper by E. Lopes that CASH_IN involves being paid by a merchant (whose name is prefixed by 'M'). 
# However, as shown below, the present data does not have merchants making CASH_IN transactions to customers.

#Are there any merchants among originator accounts for CASH_IN transactions?
head(paySim_small)
paySim_small[which((paySim_small$type == "CASH_IN") & grepl("M", paySim_small$nameOrig)), ]
# Empty data.table (0 rows) of 11 cols

# Similarly, it was stated that CASH_OUT involves paying a merchant. 
# However, for CASH_OUT transactions there are no merchants among the destination accounts.

#Are there any merchants among destination accounts for CASH_OUT transactions?
paySim_small[which((paySim_small$type == "CASH_OUT") & grepl("M", paySim_small$nameDest)), ]
#Empty data.table (0 rows) of 11 cols

# In fact, there are no merchants among any originator accounts.
# Merchants are only present in destination accounts for all PAYMENTS.

# Are there merchants among any originator accounts?
paySim_small[which(grepl("M", paySim_small$nameOrig)), ]
# Empty data.table (0 rows) of 11 cols

#Are there any transactions having merchants among destination accounts\
# other than the PAYMENT type?
paySim_small[which((paySim_small$type != "PAYMENT") & grepl("M", paySim_small$nameDest)), ]
# Empty data.table (0 rows) of 11 cols

# Conclusion: 
# Among the account labels nameOrig and nameDest, for all transactions, the merchant prefix of 'M' occurs in an unexpected way.


# Within fraudulent transactions, are there destinations for TRANSFERS 
# that are also originators for CASH_OUTs?
any(fraud_transfer$nameOrig == fraud_transfer$nameDest)
# FALSE

# Fraudulent TRANSFERs whose destination accounts are originators of \
# genuine CASH_OUTs:

fraud_transfer_dest <- data.frame(paySim[which((paySim$type == "TRANSFER") & (paySim$isFraud == 1)), ]$nameDest)
colnames(fraud_transfer_dest) <- "code"
nofraud_cashout_orig <- data.frame(paySim[which((paySim$type == "CASH_OUT") & (paySim$isFraud == 0)), ]$nameOrig)
colnames(nofraud_cashout_orig) <- "code"
joinned <- merge(fraud_transfer_dest,
                 nofraud_cashout_orig)

paySim[which((paySim$isFraud == 1) & (paySim$nameDest %in% joinned$code)), ]
paySim[which((paySim$isFraud == 0) & (paySim$nameOrig %in% joinned$code)), ]

# > paySim[which((paySim$isFraud == 1) & (paySim$nameDest %in% joinned$code)), ]
# step     type    amount    nameOrig oldbalanceOrg newbalanceOrig    nameDest oldbalanceDest newbalanceDest isFraud isFlaggedFraud
# 1:   65 TRANSFER 1282971.6 C1175896731     1282971.6              0 C1714931087              0              0       1              0
# 2:  486 TRANSFER  214793.3 C2140495649      214793.3              0  C423543548              0              0       1              0
# 3:  738 TRANSFER  814689.9 C2029041842      814689.9              0 C1023330867              0              0       1              0
# > paySim[which((paySim$isFraud == 0) & (paySim$nameOrig %in% joinned$code)), ]
# step     type    amount    nameOrig oldbalanceOrg newbalanceOrig    nameDest oldbalanceDest newbalanceDest isFraud isFlaggedFraud
# 1:  132 CASH_OUT  29084.28 C1023330867         51999       22914.72 C1422447255           0.00       29084.28       0              0
# 2:  185 CASH_OUT 214555.85  C423543548             0           0.00 C1066927674     4575179.83     4789735.69       0              0
# 3:  546 CASH_OUT  18091.05 C1714931087        197227      179135.95 C1339132632       66177.84       84268.89       0              0

# Analysis data
analysis_data_big <- paySim[which(paySim$type == "TRANSFER" | paySim$type == "CASH_OUT"), ]
analysis_data_small <- paySim_small[which(paySim_small$type == "TRANSFER" | paySim_small$type == "CASH_OUT"), ]
rm(paySim)
#drop irrelevant columns
analysis_data_big <- analysis_data_big[, -c("nameOrig", "nameDest", "isFlaggedFraud")]
analysis_data_small <- analysis_data_small[, -c("nameOrig", "nameDest", "isFlaggedFraud")]

analysis_data_big$type <- as.factor(analysis_data_big$type)
analysis_data_small$type <- as.factor(analysis_data_small$type)

# The fraction of fraudulent transactions with 'oldBalanceDest' = 'newBalanceDest' = 0 
# although the transacted 'amount' is non-zero is:
nrow(analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0) & analysis_data_big$isFraud == 1), ])/nrow(analysis_data_big[which(analysis_data_big$isFraud == 1), ])
# 0.4955558
nrow(analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0) & analysis_data_small$isFraud == 1), ])/nrow(analysis_data_small[which(analysis_data_small$isFraud == 1), ])
# 0.5

# The fraction of genuine transactions with 'oldBalanceDest' = newBalanceDest' = 0 
# although the transacted 'amount' is non-zero is:
nrow(analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0) & analysis_data_big$isFraud == 0), ])/nrow(analysis_data_big[which(analysis_data_big$isFraud == 0), ])
# 0.0006176245
nrow(analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0) & analysis_data_small$isFraud == 0), ])/nrow(analysis_data_small[which(analysis_data_small$isFraud == 0), ])
# 0.0005996587

analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0)), ]$newbalanceDest <- (-1)
analysis_data_big[which(((analysis_data_big$oldbalanceDest == 0 & analysis_data_big$newbalanceDest == 0) & analysis_data_big$amount != 0)), ]$oldbalanceDest <- (-1)

analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0)), ]$newbalanceDest <- (-1)
analysis_data_small[which(((analysis_data_small$oldbalanceDest == 0 & analysis_data_small$newbalanceDest == 0) & analysis_data_small$amount != 0)), ]$oldbalanceDest <- (-1)



analysis_data_big[which(((analysis_data_big$oldbalanceOrg == 0 & analysis_data_big$newbalanceOrg == 0) & analysis_data_big$amount != 0)), ]$newbalanceOrg <- (-1)
analysis_data_big[which(((analysis_data_big$oldbalanceOrg == 0 & analysis_data_big$newbalanceOrg == 0) & analysis_data_big$amount != 0)), ]$oldbalanceOrg <- (-1)

analysis_data_small[which(((analysis_data_small$oldbalanceOrg == 0 & analysis_data_small$newbalanceOrg == 0) & analysis_data_small$amount != 0)), ]$newbalanceOrg <- (-1)
analysis_data_small[which(((analysis_data_small$oldbalanceOrg == 0 & analysis_data_small$newbalanceOrg == 0) & analysis_data_small$amount != 0)), ]$oldbalanceOrg <- (-1)

rm(paySim_small)
rm(fraud_transfer_dest)
rm(nofraud_cashout_orig)
rm(joinned)
rm(frauds)
rm(nofraud_transfer)
rm(nofraud_cashout)
rm(fraud_transfer)
rm(nofrauds)
rm(fraud_cashout)
# Motivated by the possibility of zero-balances serving to differentiate between fraudulent and genuine transactions, 
# we take the data-imputation a step further and create 2 new features (columns) recording errors in 
# the originating and destination accounts for each transaction. 

analysis_data_big$errorBalanceOrig <- analysis_data_big$newbalanceOrig + analysis_data_big$amount - analysis_data_big$oldbalanceOrg
analysis_data_big$errorBalanceDest <- analysis_data_big$oldbalanceDest + analysis_data_big$amount - analysis_data_big$newbalanceDest

analysis_data_small$errorBalanceOrig <- analysis_data_small$newbalanceOrig + analysis_data_small$amount - analysis_data_small$oldbalanceOrg
analysis_data_small$errorBalanceDest <- analysis_data_small$oldbalanceDest + analysis_data_small$amount - analysis_data_small$newbalanceDest


# ggplot(analysis_data_big, aes(x = isFraud, y = step, color = type)) +
#   geom_jitter()

ggplot(analysis_data_small, aes(x = isFraud, y = step, color = type)) +
  geom_jitter()

ggplot(analysis_data_small, aes(x = isFraud, y = amount, color = type)) +
  geom_jitter()

ggplot(analysis_data_small, aes(x = isFraud, y = (-errorBalanceDest), color = type)) +
  geom_jitter()

plot_ly(analysis_data_small, 
        x = ~errorBalanceDest,
        y = ~(-log10(errorBalanceOrig)),
        z = ~step,
        color = ~isFraud,
        colors = c('#BF382A', '#0C4B8E'))

# converting to binary operators
analysis_data_small$type <- ifelse(analysis_data_small$type == "TRANSFER", 0, 1)

small_heatmap_nonfraud <- melt(cor(analysis_data_small[which(analysis_data_small$isFraud == 0), -c("step", "isFraud")]))
small_heatmap_fraud <- melt(cor(analysis_data_small[which(analysis_data_small$isFraud == 1), -c("step", "isFraud")]))

ggplot(small_heatmap_nonfraud, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile()

ggplot(small_heatmap_fraud, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile()

set.seed(434)
split = sample.split(analysis_data_small$isFraud, SplitRatio = 0.6)

paySim_train <- subset(analysis_data_small, split == TRUE)
paySim_train$isFraud <- as.factor(paySim_train$isFraud)
paySim_train$type<-as.factor(paySim_train$type)
paySim_train <- paySim_train[, -c("step")]

paySim_test <- subset(analysis_data_small, split == FALSE)
paySim_test$isFraud <- as.factor(paySim_test$isFraud)
paySim_test$type<-as.factor(paySim_test$type)
paySim_test <- paySim_test[, -c("step")]

paySim_train_matrix <- as.matrix(sapply(paySim_train[, -c("step")], as.numeric))

parametersGrid <-  expand.grid(eta = 0.1, 
                               colsample_bytree=c(0.5,0.7),
                               max_depth=c(3,6),
                               nrounds=100,
                               gamma=1,
                               min_child_weight=2)

ctrl_paySim <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE,
                           verboseIter = TRUE)

feature.names=names(paySim_train)
for (f in feature.names) {
  if (class(paySim_train[[f]])=="factor") {
    levels <- unique(c(paySim_train[[f]]))
    paySim_train[[f]] <- factor(paySim_train[[f]],
                         labels=make.names(levels))
  }
}
feature.names2=names(paySim_test)
for (f in feature.names2) {
  if (class(paySim_test[[f]])=="factor") {
    levels <- unique(c(paySim_test[[f]]))
    paySim_test[[f]] <- factor(paySim_test[[f]],
                                labels=make.names(levels))
  }
}

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
paySim_xgboost <- train(isFraud ~ .,
                        data = paySim_train,
                        method = "xgbTree",
                        verbose = FALSE,
                        metric = "ROC",
                        trControl = ctrl_paySim)

stopCluster(cluster)
registerDoSEQ()

xgboost_results <- predict(paySim_xgboost, newdata = paySim_test)
confusionMatrix(xgboost_results, paySim_test$isFraud)

paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "isFraud"])
}

paySim_xgboost %>%
  paySim_test_roc(data = paySim_test) %>%
  auc()




