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
library(coefplot)
options(scipen=999)

set.seed(48)

# Loading the features and the classes
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
# 0       1 
# 0.97346 0.02654 

# Getting only those customers that appear more than one
ucsd_data <- ucsd_data %>%
  dplyr::group_by(custAttr1) %>%
  dplyr::summarise(freq = n()) %>%
  dplyr::filter(freq > 1) %>%
  dplyr::inner_join(ucsd_data, by = "custAttr1") %>%
  dplyr::select(-freq)

split = sample.split(ucsd_data$Class, SplitRatio = 0.6)
ucsd_train <- subset(ucsd_data, split == TRUE)
ucsd_test <- subset(ucsd_data, split == FALSE)

feature.names=names(ucsd_train)
for (f in feature.names) {
  if (class(ucsd_train[[f]])=="factor") {
    levels <- unique(c(ucsd_train[[f]]))
    ucsd_train[[f]] <- factor(ucsd_train[[f]],
                              labels=make.names(levels))
  }
}
feature.names2=names(ucsd_test)
for (f in feature.names2) {
  if (class(ucsd_test[[f]])=="factor") {
    levels <- unique(c(ucsd_test[[f]]))
    ucsd_test[[f]] <- factor(ucsd_test[[f]],
                             labels=make.names(levels))
  }
}

rm(ucsd_data)

prop.table(table(ucsd_train$Class))
# X1         X2 
# 0.97075476 0.02924524 
prop.table(table(ucsd_test$Class))
# X1         X2 
# 0.97079489 0.02920511 

ctrl_ucsd <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 1,
                          summaryFunction = twoClassSummary,
                          #allowParallel = TRUE,
                          classProbs = TRUE,
                          verboseIter = TRUE
)


ucsd_glm <- train(Class ~ .,
                  data = ucsd_train,
                  method = "glm",
                  metric = "ROC", 
                  trControl = ctrl_ucsd)


######################################### GBM PREDICTIONS
glm_results <- predict(ucsd_glm, newdata = ucsd_test)
conf_matr_glm <- confusionMatrix(glm_results, ucsd_test$Class)
conf_matr_glm

coefplot(ucsd_glm, intercept = FALSE, color = "black")

trellis.par.set(caretTheme())
glm_imp <- varImp(ucsd_glm)
plot(glm_imp)

####################################### sampled-down model
ctrl_ucsd$sampling <- "down"

ucsd_glm_down_fit <- train(Class ~ .,
                           data = ucsd_train,
                           method = "glm",
                           metric = "ROC",
                           trControl = ctrl_ucsd)


glm_results_down <- predict(ucsd_glm_down_fit, newdata = ucsd_test)
conf_matr_glm_down <- confusionMatrix(glm_results_down, ucsd_test$Class)
conf_matr_glm_down

coefplot(ucsd_glm_down_fit, intercept = FALSE, color = "black")

glm_imp_down <- varImp(ucsd_glm_down_fit, scale = FALSE)
plot(glm_imp_down)

############# sampled-up
ctrl_ucsd$sampling <- "up"
ucsd_glm_up_fit <- train(Class ~ .,
                         data = ucsd_train,
                         method = "glm",
                         metric = "ROC",
                         trControl = ctrl_ucsd)

glm_results_up <- predict(ucsd_glm_up_fit, newdata = ucsd_test)
conf_matr_glm_up <- confusionMatrix(glm_results_up, ucsd_test$Class)
conf_matr_glm_up

coefplot(ucsd_glm_up_fit, intercept = FALSE, color = "black")

glm_imp_up <- varImp(ucsd_glm_up_fit, scale = FALSE)
plot(glm_imp_up)


############# SMOTE
ctrl_ucsd$sampling <- "smote"
ucsd_glm_smote_fit <- train(Class ~ .,
                            data = ucsd_train,
                            method = "glm",
                            metric = "ROC",
                            trControl = ctrl_ucsd)

glm_results_smote <- predict(ucsd_glm_smote_fit, newdata = ucsd_test)
conf_matr_glm_smote <- confusionMatrix(glm_results_smote, ucsd_test$Class)
conf_matr_glm_smote

coefplot(ucsd_glm_smote_fit, intercept = FALSE, color = "black")

glm_imp_smote <- varImp(ucsd_glm_smote_fit, scale = FALSE)
plot(glm_imp_smote)

####################################################################
# ROC and AUC
ucsd_test_roc <- function(model, data) {
  roc(data$Class,
      predict(model, data, type = "prob")[, "X2"])
}

ucsd_glm_model_list <- list(original = ucsd_glm,
                            down = ucsd_glm_down_fit,
                            up = ucsd_glm_up_fit,
                            SMOTE = ucsd_glm_smote_fit)


ucsd_glm_model_list_roc <- ucsd_glm_model_list %>%
  map(ucsd_test_roc, data = ucsd_test)

ucsd_auc_glm <- as.data.frame(ucsd_glm_model_list_roc %>% map(auc))
saveRDS(ucsd_auc_glm, 
        file = paste0(getwd(),"/figures/ucsd/glm/ucsd_auc_glm.rds"))

ucsd_glm_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in ucsd_glm_model_list_roc){
  ucsd_glm_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(ucsd_glm_model_list)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_glm_results_df_roc <- bind_rows(ucsd_glm_results_list_roc)
saveRDS(ucsd_glm_results_df_roc, 
        file = paste0(getwd(),"/figures/ucsd/glm/ucsd_glm_results_df_roc.rds"))


custom_col <- c("#000000", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = ucsd_glm_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)

####  Construction the precision/recall graphic
ucsd_glm_calc_auprc <- function(model, data) {
  index_class2 <- data$Class == "X2"
  index_class1 <- data$Class == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

ucsd_glm_model_list_pr <- ucsd_glm_model_list %>%
  map(ucsd_glm_calc_auprc, data = ucsd_test)

# Precision recall Curve AUC calculation
ucsd_PR_glm <- as.data.frame(ucsd_glm_model_list_pr %>% map(function(the_mod) the_mod$auc.integral))
saveRDS(ucsd_PR_glm, 
        file = paste0(getwd(),"/figures/ucsd/glm/ucsd_PR_glm.rds"))


ucsd_glm_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in ucsd_glm_model_list_pr) {
  ucsd_glm_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(ucsd_glm_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

ucsd_glm_results_df_pr <- bind_rows(ucsd_glm_results_list_pr)
saveRDS(ucsd_glm_results_df_pr, 
        file = paste0(getwd(),"/figures/ucsd/glm/ucsd_glm_results_df_pr.rds"))


ggplot(aes(x = recall, y = precision, group = model), data = ucsd_glm_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(ucsd_test$Class == "X2")/nrow(ucsd_test),slope = 0, color = "gray", size = 1)
