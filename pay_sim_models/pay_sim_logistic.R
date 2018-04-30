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
library(brglm)
options(scipen=999)

set.seed(48)
paySim <- fread("C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/pay_sim_synthetic/PS_20174392719_1491204439457_log.csv",
                header = TRUE,
                sep = ",")
paySim_small <- paySim[sample(nrow(paySim), 50000), ] 

# Fraud Rate
prop.table(table(paySim_small$isFraud))

# Where does fraud occur -> only in CASH_OUT AND TRANSFER type of payments
plyr::count(paySim_small, c("type", "isFraud"))

fraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 1)), ]
fraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 1)), ]

nofraud_transfer <- paySim_small[which((paySim_small$type == "TRANSFER") & (paySim_small$isFraud == 0)), ]
nofraud_cashout <- paySim_small[which((paySim_small$type == "CASH_OUT") & (paySim_small$isFraud == 0)), ]

frauds <- paySim_small[which(paySim_small$isFraud == 1), ] 
nofrauds <- paySim_small[which(paySim_small$isFraud == 0), ] 



fraud_transfer_dest <- data.frame(paySim[which((paySim$type == "TRANSFER") & (paySim$isFraud == 1)), ]$nameDest)
colnames(fraud_transfer_dest) <- "code"
nofraud_cashout_orig <- data.frame(paySim[which((paySim$type == "CASH_OUT") & (paySim$isFraud == 0)), ]$nameOrig)
colnames(nofraud_cashout_orig) <- "code"
joinned <- merge(fraud_transfer_dest,
                 nofraud_cashout_orig)


# Analysis data
analysis_data_big <- paySim[which(paySim$type == "TRANSFER" | paySim$type == "CASH_OUT"), ]
analysis_data_small <- paySim_small[which(paySim_small$type == "TRANSFER" | paySim_small$type == "CASH_OUT"), ]
rm(paySim)
#drop irrelevant columns
analysis_data_big <- analysis_data_big[, -c("nameOrig", "nameDest", "isFlaggedFraud")]
analysis_data_small <- analysis_data_small[, -c("nameOrig", "nameDest", "isFlaggedFraud")]

analysis_data_big$type <- as.factor(analysis_data_big$type)
analysis_data_small$type <- as.factor(analysis_data_small$type)



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

rm(analysis_data_big)

ctrl_paySim <- trainControl(method = "repeatedcv",
                            number = 10,
                            repeats = 1,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = TRUE)

#paySim_single <- glm(isFraud ~., family = binomial, data = paySim_train)
paySim_single_biasreduce <- brglm(isFraud ~., family = binomial, data = paySim_train)


modelInfo <- list(label = "Bias Reduced GLM",
                  library = c("brglm"),
                  type = "Classification",
                  loop = NULL,
                  parameters = data.frame(parameter = "parameter",
                                          class = "character",
                                          label = "parameter"),
                  grid = function(x, y, len = NULL, search = "grid") data.frame(parameter = "none"),
                  fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
                    dat <- if(is.data.frame(x)) x else as.data.frame(x)
                    dat$.outcome <- y
                    if(length(levels(y)) > 2) stop("brglm models can only use 2-class outcomes")
                    theDots <- list(...)
                    if(!any(names(theDots) == "family"))
                    {
                      theDots$family <- if(is.factor(y)) binomial() else gaussian()
                    }
                    ## pass in any model weights
                    if(!is.null(wts)) theDots$weights <- wts
                    
                    modelArgs <- c(list(formula = as.formula(".outcome ~ ."), data = dat), theDots)
                    
                    out <- do.call("brglm", modelArgs)
                    ## When we use do.call(), the call infformation can contain a ton of
                    ## information. Inlcuding the contenst of the data. We eliminate it.
                    out$call <- NULL
                    out
                  },
                  predict = function(modelFit, newdata, submodels = NULL) {
                    if(!is.data.frame(newdata)) newdata <- as.data.frame(newdata)
                    if(modelFit$problemType == "Classification") {
                      probs <-  predict(modelFit, newdata, type = "response")
                      out <- ifelse(probs < .5,
                                    modelFit$obsLevel[1],
                                    modelFit$obsLevel[2])
                    } else {
                      out <- predict(modelFit, newdata, type = "response")
                    }
                    out
                  },
                  prob = function(modelFit, newdata, submodels = NULL){
                    if(!is.data.frame(newdata)) newdata <- as.data.frame(newdata)
                    out <- predict(modelFit, newdata, type = "response")
                    out <- cbind(1-out, out)
                    ## glm models the second factor level, we treat the first as the
                    ## event of interest. See Details in ?glm
                    dimnames(out)[[2]] <-  modelFit$obsLevels
                    out
                  },
                  varImp = function(object, ...) {
                    values <- summary(object)$coef
                    varImps <-  abs(values[-1, grep("value$", colnames(values)), drop = FALSE])
                    vimp <- data.frame(varImps)
                    colnames(vimp) <- "Overall"
                    if(!is.null(names(varImps))) rownames(vimp) <- names(varImps)
                    vimp
                  },
                  predictors = function(x, ...) predictors(x$terms),
                  levels = function(x) if(any(names(x) == "obsLevels")) x$obsLevels else NULL,
                  tags = c("Bias Reduced Logistic Regression"),
                  sort = function(x) x)

paySim_brglm <- train(isFraud ~ .,
                    data = paySim_train,
                    method = modelInfo,
                    metric = "ROC", 
                    maxit = 10000000,
                    trControl = ctrl_paySim)

brglm_results <- predict(paySim_brglm, newdata = paySim_test)
confusionMatrix(brglm_results, paySim_test$isFraud)

coefplot(paySim_brglm, intercept = FALSE, color = "black")

trellis.par.set(caretTheme())
brglm_imp <- varImp(paySim_brglm, scale = FALSE)
plot(brglm_imp)

############### sampled-down model
ctrl_paySim$seeds <- paySim_glm$control$seeds

ctrl_paySim$sampling <- "down"

paySim_glm_down_fit <- train(isFraud ~ .,
                             data = paySim_train,
                             method = modelInfo,
                             maxit = 10000000,
                             metric = "ROC",
                             trControl = ctrl_paySim)

### Sampled-down fit
glm_down_results <- predict(paySim_glm_down_fit, newdata = paySim_test)
confusionMatrix(glm_down_results, paySim_test$isFraud)

coefplot(paySim_glm_down_fit, intercept = FALSE, color = "black")

glm_down_imp <- varImp(paySim_glm_down_fit, scale = FALSE)
plot(glm_down_imp)

############# sampled-up
ctrl_paySim$sampling <- "up"

paySim_glm_up_fit <- train(isFraud ~ .,
                           data = paySim_train,
                           method = modelInfo,
                           maxit = 10000000,
                           metric = "ROC",
                           trControl = ctrl_paySim)


### Sampled-up fit
glm_up_results <- predict(paySim_glm_up_fit, newdata = paySim_test)
confusionMatrix(glm_up_results, paySim_test$isFraud)

coefplot(paySim_glm_up_fit, intercept = FALSE, color = "black")

glm_up_imp <- varImp(paySim_glm_up_fit, scale = FALSE)
plot(glm_up_imp)

############# SMOTE
ctrl_paySim$sampling <- "smote"

paySim_glm_smote_fit <- train(isFraud ~ .,
                              data = paySim_train,
                              method = modelInfo,
                              maxit = 10000000,
                              metric = "ROC",
                              trControl = ctrl_paySim)


### Smote fit
glm_smote_results <- predict(paySim_glm_smote_fit, newdata = paySim_test)
confusionMatrix(glm_smote_results, paySim_test$isFraud)

coefplot(paySim_glm_smote_fit, intercept = FALSE, color = "black")

glm_smote_imp <- varImp(paySim_glm_smote_fit, scale = FALSE)
plot(glm_smote_imp)

####################################################
paySim_test_roc <- function(model, data) {
  roc(data$isFraud,
      predict(model, data, type = "prob")[, "X2"])
}

paySim_glm_model_list <- list(original = paySim_brglm,
                              down = paySim_glm_down_fit,
                              up = paySim_glm_up_fit,
                              SMOTE = paySim_glm_smote_fit)


paySim_glm_model_list_roc <- paySim_glm_model_list %>%
  map(paySim_test_roc, data = paySim_test)

paySim_auc_glm <- as.data.frame(paySim_glm_model_list_roc %>% map(auc))
saveRDS(paySim_auc_glm, 
        file = paste0(getwd(),"/figures/paysim/glm/paysim_auc_glm.rds"))

paySim_glm_results_list_roc <- list(NA)
num_mod <- 1

for(the_roc in paySim_glm_model_list_roc){
  paySim_glm_results_list_roc[[num_mod]] <-
    data_frame(tpr = the_roc$sensitivities,
               fpr = 1 - the_roc$specificities,
               model = names(paySim_glm_model_list)[num_mod])
  num_mod <- num_mod + 1
}

paySim_glm_results_df_roc <- bind_rows(paySim_glm_results_list_roc)
saveRDS(paySim_glm_results_df_roc, 
        file = paste0(getwd(),"/figures/paysim/glm/paySim_glm_results_df_roc.rds"))

custom_col <- c("#000000", "#0072B2", "#D55e00", "#CC79A7")

ggplot(aes(x = fpr, y = tpr, group = model), data = paySim_glm_results_df_roc) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = 0, slope = 1, color = "gray", size = 1) +
  theme_bw(base_size = 18)


####  Construction the precision/recall graphic
paySim_glm_calc_auprc <- function(model, data) {
  index_class2 <- data$isFraud == "X2"
  index_class1 <- data$isFraud == "X1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$X2[index_class2],
           predictions$X2[index_class1],
           curve = TRUE)
}

paySim_glm_model_list_pr <- paySim_glm_model_list %>%
  map(paySim_glm_calc_auprc, data = paySim_test)

# Precision recall Curve AUC calculation
paySim_PR_glm <- as.data.frame(paySim_glm_model_list_pr %>% map(function(the_mod) the_mod$auc.integral))
saveRDS(paySim_PR_glm, 
        file = paste0(getwd(),"/figures/paysim/glm/paySim_PR_glm.rds"))

paySim_glm_results_list_pr <- list(NA)
num_mod <- 1
for (the_pr in paySim_glm_model_list_pr) {
  paySim_glm_results_list_pr[[num_mod]] <-
    data_frame(recall = the_pr$curve[, 1],
               precision = the_pr$curve[, 2],
               model = names(paySim_glm_model_list_pr)[num_mod])
  num_mod <- num_mod + 1
}

paySim_glm_results_df_pr <- bind_rows(paySim_glm_results_list_pr)
saveRDS(paySim_glm_results_df_pr, 
        file = paste0(getwd(),"/figures/paysim/glm/paySim_glm_results_df_pr.rds"))

ggplot(aes(x = recall, y = precision, group = model), data = paySim_glm_results_df_pr) +
  geom_line(aes(color = model), size = 1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept = sum(paySim_test$Class == "X2")/nrow(paySim_test),slope = 0, color = "gray", size = 1)

