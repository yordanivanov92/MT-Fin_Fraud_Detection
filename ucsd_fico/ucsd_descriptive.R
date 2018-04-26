library(ggplot2)
library(GGally)
library(caTools)
library(plyr)
library(plotly)
options(scipen=999)

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

ucsd_data <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1) %>%
  dplyr::inner_join(ucsd_data, by = "custAttr1") %>%
  dplyr::select(-c(freq, custAttr1, zip1)) 


table(ucsd_data$Class)


ggplot(ucsd_data, aes(x = Class, y = amount, group = Class)) +
  geom_boxplot() +
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud"))

ggplot(ucsd_data[ucsd_data$amount < 40,], aes(x = Class, y = amount, group = Class)) +
  geom_boxplot() +
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud"))

ggplot(ucsd_data[ucsd_data$amount < 25,], aes(x = Class, y = amount, group = Class)) +
  geom_boxplot() +
  scale_x_discrete(breaks = c(0,1),
                   labels = c("No Fraud", "Fraud"))


ggplot(ucsd_data, aes(x = field1, fill = Class)) +
  geom_histogram(bins = 4, alpha=.5, position="identity")

ggplot(ucsd_data, aes(x = field2, fill = Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = field3, fill = Class)) +
  geom_histogram(bins = 50, alpha=.5, position="identity")

ggplot(ucsd_data, aes(x = field4, fill = Class)) +
  geom_histogram(bins = 30, alpha=.5, position="identity")


ggplot(ucsd_data, aes(x = flag1 , fill=Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = flag2 , fill=Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = flag3 , fill=Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = flag4 , fill=Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = flag5 , fill=Class)) +
  geom_histogram(bins = 13, alpha=.5, position="identity")

ggplot(ucsd_data, aes(x = indicator1 , fill=Class)) +
  geom_bar()

ggplot(ucsd_data, aes(x = indicator2 , fill=Class)) +
  geom_bar()
