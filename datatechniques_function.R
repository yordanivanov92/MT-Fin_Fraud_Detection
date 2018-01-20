

paySim_xgboost <- train(isFraud ~ .,
                        data = paySim_train,
                        method = "xgbTree",
                        verbose = FALSE,
                        metric = "ROC", 
                        trControl = ctrl_paySim)




paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "X2"])
}

paySim_xgboost %>%
  paySim_test_roc(data = paySim_test) %>%
  auc()

### Original Fit
xgboost_results <- predict(paySim_xgboost, newdata = paySim_test)
confusionMatrix(xgboost_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgboost, metric = "ROC")

xgboost_imp <- varImp(paySim_xgboost, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_imp)

################## COST SENSITIVE XGBOOST MODEL
# The penalization costs can be tinkered with
paySim_model_weights <- ifelse(paySim_train$isFraud == "X1",
                               (1/table(paySim_train$isFraud)[1]) * 0.5,
                               (1/table(paySim_train$isFraud)[2]) * 0.5)

ctrl_paySim$seeds <- paySim_xgboost$control$seeds

 # convention to leave 1 core for OS

paySim_xgboost_weighted_fit <- train(isFraud ~ .,
                                     data = paySim_train,
                                     method = "xgbTree",
                                     verbose = FALSE,
                                     weights = paySim_model_weights,
                                     metric = "ROC", 
                                     trControl = ctrl_paySim)




############### sampled-down model
ctrl_paySim$sampling <- "down"
 # convention to leave 1 core for OS

paySim_xgboost_down_fit <- train(isFraud ~ .,
                                 data = paySim_train,
                                 method = "xgbTree",
                                 verbose = FALSE,
                                 metric = "ROC",
                                 trControl = ctrl_paySim)



############# sampled-up
ctrl_paySim$sampling <- "up"
 # convention to leave 1 core for OS

paySim_xgboost_up_fit <- train(isFraud ~ .,
                               data = paySim_train,
                               method = "xgbTree",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl_paySim)



############# SMOTE
ctrl_paySim$sampling <- "smote"
 # convention to leave 1 core for OS

paySim_xgboost_smote_fit <- train(isFraud ~ .,
                                  data = paySim_train,
                                  method = "xgbTree",
                                  verbose = FALSE,
                                  metric = "ROC",
                                  trControl = ctrl_paySim)



paySim_xgboost_model_list <- list(original = paySim_xgboost,
                                  weighted = paySim_xgboost_weighted_fit,
                                  down = paySim_xgboost_down_fit,
                                  up = paySim_xgboost_up_fit,
                                  SMOTE = paySim_xgboost_smote_fit)
paySim_xgboost_model_list_roc <- paySim_xgboost_model_list %>%
  map(paySim_test_roc, data = paySim_train)

paySim_xgboost_model_list_roc %>%
  map(auc)

paySim_xgboost_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in paySim_xgboost_model_list_roc){
  paySim_xgboost_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(paySim_xgboost_model_list)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgboost_results_df_roc <- bind_rows(paySim_xgboost_results_list_roc)

custom_col <- c("#000000", "#009E73", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_xgboost_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
paySim_xgboost_calc_auprc <- function(model, data) {
  index_class2 <- data$type == "X2"
  index_class1 <- data$type == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$type[index_class2],
           predictions$type[index_class1],
           curve = TRUE)
}

#### ERROR HERE - FIX
paySim_xgboost_model_list_pr <- paySim_xgboost_model_list %>%
  map(paySim_xgboost_calc_auprc, data = paySim_test)


paySim_xgboost_model_list_pr %>%
  map(function(the_mod) the_mod$auc.integral)

paySim_xgboost_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in paySim_xgboost_model_list_pr) {
  paySim_xgboost_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(paySim_xgboost_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

paySim_xgboost_results_df_pr <- bind_rows(paySim_xgboost_results_list_pr)

ggplot(aes(x = recall, y = precision, group = model), data = paySim_xgboost_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(paySim_test$type == "X2")/nrow(paySim_test),slope = 0, color = "gray", size = 1)


##### HAVE ANOTHER LOOK HERE - NOT ADAPTED
paySim_xgboostSim_auprcSummary <- function(data, lev = NULL, model = NULL){
  
  index_class2 <- data$obs == "X2"
  index_class1 <- data$obs == "X1"
  
  the_curve <- pr.curve(data$type[index_class2],
                        data$type[index_class1],
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
                 method = "xgbTree",
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




################### Results and some graphs
### Original Fit
xgboost_results <- predict(paySim_xgboost, newdata = paySim_test)
confusionMatrix(xgboost_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_xgboost, metric = "ROC")

xgboost_imp <- varImp(paySim_xgboost, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_imp)

### Weighted fit
xgboost_weight_results <- predict(paySim_xgboost_weighted_fit, newdata = paySim_test)
confusionMatrix(xgboost_weight_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_weight_xgboost, metric = "ROC")

xgboost_weight_imp <- varImp(paySim_xgboost_weighted_fit, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_weight_imp)

### Sampled-down fit
xgboost_down_results <- predict(paySim_xgboost_down_fit, newdata = paySim_test)
confusionMatrix(xgboost_down_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_down_xgboost, metric = "ROC")

xgboost_down_imp <- varImp(paySim_xgboost_down_fit, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_down_imp)

### Sampled-up fit
xgboost_up_results <- predict(paySim_xgboost_up_fit, newdata = paySim_test)
confusionMatrix(xgboost_up_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_up_xgboost, metric = "ROC")

xgboost_up_imp <- varImp(paySim_xgboost_up_fit, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_up_imp)

### Smote fit
xgboost_smote_results <- predict(paySim_xgboost_smote_fit, newdata = paySim_test)
confusionMatrix(xgboost_smote_results, paySim_test$isFraud)

trellis.par.set(caretTheme())
plot(paySim_smote_xgboost, metric = "ROC")

xgboost_smote_imp <- varImp(paySim_xgboost_smote_fit, scale = FALSE)
#xgboost_imp - variable importance is observed
plot(xgboost_smote_imp)