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
library(xgboost)
library(GGally)
options(scipen=999)

set.seed(48)

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
# 0          1 
# 0.97077081 0.02922919 

multi_obs <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1)

ucsd_data <- join(ucsd_data, multi_obs, by = "custAttr1", type = "inner") %>%
  dplyr::select(-freq)
rm(multi_obs)

ucsd_data$Class <- as.factor(ucsd_data$Class)
prop.table(table(ucsd_data$Class))

plyr::count(ucsd_data, c("zip1", "Class")) %>% dplyr::arrange(-freq) %>% dplyr::filter(Class == 1)
#     zip1 Class freq
# 1    920     1   73
# 2    600     1   57
# 3    852     1   42
# 4    100     1   36
# 5    917     1   31
# 6    891     1   29
# 7    940     1   26
# 8    921     1   20
# 9    770     1   19
# 10   945     1   17
# 11   956     1   17
# 12   430     1   16
# 13   900     1   16
# 14    88     1   15
# Checking fraud by amount
ggplot(ucsd_data, aes(x = Class, y = amount, group = Class)) +
  geom_boxplot()

ggplot(ucsd_data[ucsd_data$amount < 40,], aes(x = Class, y = amount, group = Class)) +
  geom_boxplot()

# Field3 exploration - most important variable in model
ggplot(ucsd_data, aes(x = Class, y = field3, group = Class)) +
  geom_boxplot()

ggplot(ucsd_data, aes(x = field3)) + 
  geom_histogram()

ggplot(ucsd_data[ucsd_data$Class == 0,], aes(x = field3)) + 
  geom_histogram()

ggplot(ucsd_data[ucsd_data$Class == 1,], aes(x = field3)) +
  geom_histogram()


###################
ggpairs(ucsd_data, columns = c("amount", "zip1", "field3", "Class"), mapping = aes(colour = Class))


plotly::plot_ly(ucsd_data,
                x = ~amount,
                y = ~field3,
                z = ~zip1,
                color = ~as.factor(Class),
                colors = c("yellow", "black"),
                type = "scatter3d")

ggplot(ucsd_data, aes(x = Class, y = field3, color = as.factor(zip1))) +
  geom_jitter() +
  theme(legend.position = "none")
