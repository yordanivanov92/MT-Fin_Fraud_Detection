# Handling class imbalancedness - steps from "Handling Imbalance with R and Caret"
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

bankSim <- bankSim %>%
  select(customer, age, gender, merchant, category, amount, fraud)
##############
# filtered_customers <- plyr::count(bankSim, c("customer", "fraud"))
# dupl_data_customers <- filtered[duplicated(filtered[, "customer"]), ]
# 
# #getting those customers that have exhibited fraud
# bankSim_filter_customers <- bankSim %>%
#   filter(customer %in% dupl_data[, "customer"])  %>%
#   select(customer) %>%
#   distinct(customer) %>%
#   arrange(customer)
# 
# bankSim <- bankSim_filter_customers %>%
#   filter(customer %in% bankSim_filter_customers[, "customer"])
# ###################
# filtered_merchant <- plyr::count(bankSim, c("merchant", "fraud"))
# dupl_data_customers <- filtered[duplicated(filtered[, "customer"]), ]
# 
# #getting those customers that have exhibited fraud
# bankSim_filter_customers <- bankSim %>%
#   filter(customer %in% dupl_data[, "customer"])  %>%
#   select(customer) %>%
#   distinct(customer) %>%
#   arrange(customer)
# 
# bankSim_customers_total <- bankSim %>%
#   filter(customer %in% bankSim_filter_customers[, "customer"])


# Removing
bankSim_filter <- bankSim %>%
  filter(category != "'es_transportation'") %>%
  filter(category != "'es_food'") %>%
  filter(category != "'es_contents'") %>%
  filter(gender != "'U'") %>%
  filter(merchant != "'M1053599405'") %>%
  filter(merchant != "'M117188757'") %>%
  filter(merchant != "'M1313686961'") %>%
  filter(merchant != "'M1352454843'") %>%
  filter(merchant != "'M1400236507'") %>%
  filter(merchant != "'M1416436880'") %>%
  filter(merchant != "'M1600850729'") %>%
  filter(merchant != "'M1726401631'") %>%
  filter(merchant != "'M1788569036'") %>%
  filter(merchant != "'M1823072687'") %>%
  filter(merchant != "'M1842530320'") %>%
  filter(merchant != "'M1872033263'") %>%
  filter(merchant != "'M1913465890'") %>%
  filter(merchant != "'M1946091778'") %>%
  filter(merchant != "'M348934600'") %>%
  filter(merchant != "'M349281107'") %>%
  filter(merchant != "'M45060432'") %>%
  filter(merchant != "'M677738360'") %>%
  filter(merchant != "'M85975013'") %>%
  filter(merchant != "'M97925176'")

################## This would be working towards increasing prediction power
# only in the current dataset. When new observation are being added in a 
# system, this could lead to problems, as some useful observation could be erased.
# In the end, it is genuinely, selective undersampling.
customer_fraud_freq <- plyr::count(bankSim_filter, c("customer", "fraud"))
dupl_data <- customer_fraud_freq[duplicated(customer_fraud_freq$customer), ]
#colnames(dupl_data) <- "customer"

#getting those customers that have exhibited fraud
bankSim_filter_cust <- bankSim_filter %>%
  filter(customer %in% dupl_data$customer) %>%
  select(customer) %>%
  distinct(customer) %>%
  arrange(customer)

bankSim_filter_total <- bankSim_filter %>%
  filter(customer %in% bankSim_filter_cust$customer)

bankSim <- bankSim_filter_total
plyr::count(bankSim, c("category", "fraud"))
# es_contents - no fraud
# es_food - no fraud
# es_transportation - no fraud


plyr::count(bankSim, c("gender", "fraud"))
# the U gender - no frauds commited
plyr::count(bankSim, c("merchant", "fraud"))
# some merchants - no fraud
plyr::count(bankSim, c("age", "fraud"))

# for (category in colnames(bankSim)){
#   filtered <- plyr::count(bankSim, c(category, "fraud"))
#   dupl_data <- filtered[duplicated(filtered[, category]), ]
#   #colnames(dupl_data) <- "customer"
# 
#   #getting those customers that have exhibited fraud
#   bankSim_filter_category <- bankSim %>%
#     filter(category %in% dupl_data[, category])  %>%
#     select(category) %>%
#     distinct(category) %>%
#     arrange(category)
# 
#   bankSim_filter_total <- bankSim %>%
#     filter(category %in% bankSim_filter_category[, category])
# }




split = sample.split(bankSim$fraud, SplitRatio = 0.6)

bankSim_train = subset(bankSim, split == TRUE)
bankSim_test = subset(bankSim, split == FALSE)

prop.table(table(bankSim_train$fraud))
prop.table(table(bankSim_test$fraud))

ctrl_bankSim <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = TRUE)


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

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_orig_fit <- train(fraud ~ .,
                          data = bankSim_train,
                          method = "gbm",
                          verbose = FALSE,
                          metric = "ROC",
                          trControl = ctrl_bankSim)

stopCluster(cluster)
registerDoSEQ()



bankSim_test_roc <- function(model, data) {
  roc(data$fraud,
      predict(model, data, type = "prob")[, "fraud"])
}

bankSim_orig_fit %>%
  bankSim_test_roc(data = bankSim_test) %>%
  auc()


# Handling class imbalance with weighted or sampling methods
bankSim_model_weights <- ifelse(bankSim_train$fraud == "clean",
                        (1/table(bankSim_train$fraud)[1]) * 0.5,
                        (1/table(bankSim_train$fraud)[2]) * 0.5)

ctrl_bankSim$seeds <- bankSim_orig_fit$control$seeds
#weighted model
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_weighted_fit <- train(fraud ~ .,
                      data = bankSim_train,
                      method = "gbm",
                      verbose = FALSE,
                      weights = bankSim_model_weights,
                      metric = "ROC",
                      trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()
#sampled-down model
ctrl_bankSim$sampling <- "down"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_down_fit <- train(fraud ~ .,
                  data = bankSim_train,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl_bankSim)

#sampled-up
ctrl_bankSim$sampling <- "up"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_up_fit <- train(fraud ~ .,
                data = bankSim_train,
                method = "gbm",
                verbose = FALSE,
                metric = "ROC",
                trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()
#SMOTE
ctrl_bankSim$sampling <- "smote"
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
bankSim_smote_fit <- train(fraud ~ .,
                   data = bankSim_train,
                   method = "gbm",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl_bankSim)
stopCluster(cluster)
registerDoSEQ()


bankSim_model_list <- list(original = bankSim_orig_fit,
                   weighted = bankSim_weighted_fit,
                   down = bankSim_down_fit,
                   up = bankSim_up_fit,
                   SMOTE = bankSim_smote_fit)
bankSim_model_list_roc <- bankSim_model_list %>%
  map(bankSim_test_roc, data = bankSim_train)

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


bankSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$obs == "fraud"
  index_class1 <- data$obs == "clean"
  
  the_curve <- pr.curve(data$fraud[index_class2],
                        data$fraud[index_class1],
                        curve = FALSE)
  
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}

#Re-initialize control function to remove smote and
# include our new summary function

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = auprcSummary,
                     classProbs = TRUE,
                     seeds = orig_fit$control$seeds)

orig_pr <- train(Class ~ .,
                 data = imbal_train,
                 method = "gbm",
                 verbose = FALSE,
                 metric = "AUPRC",
                 trControl = ctrl)

# Get results for auprc on the test set

orig_fit_test <- orig_fit %>%
  calc_auprc(data = imbal_test) %>%
  (function(the_mod) the_mod$auc.integral)

orig_pr_test <- orig_pr %>%
  calc_auprc(data = imbal_test) %>%
  (function(the_mod) the_mod$auc.integral)

# The test errors are the same

identical(orig_fit_test,
          orig_pr_test)
## [1] TRUE
# Because both chose the same
# hyperparameter combination

identical(orig_fit$bestTune,
          orig_pr$bestTune)



test_results_orig <- predict(bankSim_orig_fit, newdata = bankSim_test)
confusionMatrix(test_results_orig, bankSim_test$fraud, positive = "fraud")

test_results_weight <- predict(bankSim_weighted_fit, newdata = bankSim_test)
confusionMatrix(test_results_weight, bankSim_test$fraud, positive = "fraud")

test_results_down <- predict(bankSim_down_fit, newdata = bankSim_test)
confusionMatrix(test_results_down, bankSim_test$fraud, positive = "fraud")

test_results_up <- predict(bankSim_up_fit, newdata = bankSim_test)
confusionMatrix(test_results_up, bankSim_test$fraud, positive = "fraud")

test_results_smote <- predict(bankSim_smote_fit, newdata = bankSim_test)
conf_smote <- confusionMatrix(test_results_smote, bankSim_test$fraud, positive = "fraud")


