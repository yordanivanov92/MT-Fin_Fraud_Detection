library(ggplot2) 
library(readr)
library(dplyr)
library(DT)
library(ggthemes)
library(randomForest)
library(InformationValue)

bankSim_RF_big <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                       header = TRUE,
                       sep = ",")
bankSim_RF <- bankSim_RF_big[sample(nrow(bankSim_RF_big), 10000), ]

bankSim_RF$step <- factor(bankSim_RF$step)
bankSim_RF$customer <- factor(bankSim_RF$customer)
bankSim_RF$age <- factor(bankSim_RF$age)
bankSim_RF$gender <- factor(bankSim_RF$gender)
bankSim_RF$zipcodeOri <- factor(bankSim_RF$zipcodeOri)
bankSim_RF$merchant <- factor(bankSim_RF$merchant)
bankSim_RF$zipMerchant <- factor(bankSim_RF$zipMerchant)
bankSim_RF$category <- factor(bankSim_RF$category)
bankSim_RF$fraud <- factor(bankSim_RF$fraud)

summary(bankSim_RF)

frauddd1 <- bankSim_RF %>%
  group_by(as.factor(bankSim_RF$amount)) %>%
  dplyr::summarize(Total = n())

frauddd2 <- bankSim_RF %>%
  group_by(as.factor(bankSim_RF$age)) %>%
  dplyr::summarize(Total = n())

frauddd3 <- bankSim_RF %>%
  group_by(as.factor(bankSim_RF$gender)) %>%
  dplyr::summarize(Total = n())

frauddd4 <- bankSim_RF %>%
  group_by(as.factor(bankSim_RF$category)) %>%
  dplyr::summarize(Total = n())

frauddd5 <- bankSim_RF %>%
  group_by(as.factor(bankSim_RF$step)) %>%
  dplyr::summarize(Total = n())


ggplot(frauddd1, aes(frauddd1$`as.factor(bankSim_RF$amount)`, Total,fill = I("#fec106"))) + 
  geom_bar( stat = "identity",colour="#FF9999") + 
  xlab("Amount") +
  ylab("Total") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(frauddd2, aes(frauddd2$`as.factor(bankSim_RF$age)`, Total,fill=I("#4699dd"))) + 
  geom_bar( stat = "identity",colour="#FF9999") + 
  xlab("Age") +
  ylab("Total")

ggplot(frauddd3, aes(frauddd3$`as.factor(bankSim_RF$gender)`, Total,fill = I("#56ddc5"))) + 
  geom_bar( stat = "identity",colour="#FF9999") + 
  xlab("Gender") + 
  ylab("Total")

ggplot(frauddd4, aes(frauddd4$`as.factor(bankSim_RF$category)`, Total,fill = I("#f89685"))) + 
  geom_bar( stat = "identity") +
  xlab("Category") + 
  ylab("Total") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(frauddd5, aes(frauddd5$`as.factor(bankSim_RF$step)`, Total,fill = I("#4699dd"))) + 
  geom_bar( stat = "identity" ) + 
  xlab("Step") +
  ylab("Total") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

dff = bankSim_RF[,c("merchant","category","gender","age","amount","fraud")]

train_sample = sample(nrow(dff), size = nrow(dff)*0.66)

train_data = dff[train_sample,]
test_data = dff[-train_sample,]

rf = randomForest(y=train_data$fraud, 
                  x = train_data[, -ncol(train_data)],
                  ytest = test_data$fraud, 
                  xtest = test_data[, -ncol(test_data)],
                  ntree = 200,
                  classwt = c(0.7, 0.3),
                  keep.forest = T)

varImpPlot(rf, type=2)
rf

partialPlot(rf, train_data,merchant , 1, las=2)
partialPlot(rf, train_data,category , 1, las=2)
partialPlot(rf, train_data,amount , 1)

bankSim_RF_cut <- bankSim_RF[bankSim_RF$amount<800, ]
bankSim_RF_cut$amount <- cut(bankSim_RF_cut$amount,
                             breaks=50,
                             dig.lab=2,
                             labels=1:50)

dff_log <- bankSim_RF_cut[which((bankSim_RF_cut$merchant != "M1294758098") & (bankSim_RF_cut$merchant != "M933210764")),
                          c("merchant","category","amount","fraud")]
train_sample = sample(nrow(dff_log), size = nrow(dff_log)*0.66)
train_data = dff[train_sample,]
test_data = dff[-train_sample,]

logitMod <- glm(fraud~ merchant+category+amount, data=train_data, family=binomial(link="logit"))
predicted <- predict(logitMod, test_data, type="response")
plotROC(test_data$fraud, predicted)


optCutOff <- optimalCutoff(test_data$fraud, predicted)[1]

misClassError(test_data$fraud, predicted, threshold = optCutOff)
sensitivity(test_data$fraud, predicted, threshold = optCutOff)
specificity(test_data$fraud, predicted, threshold = optCutOff)
