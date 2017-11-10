library(caret)
library(pROC)

set.seed(442)
trainingSet <- twoClassSim(n = 500, intercept = -16)
testingSet  <- twoClassSim(n = 500, intercept = -16)

## Class frequencies
table(trainingSet$Class)

set.seed(949)
mod0 <- train(Class ~ ., data = trainingSet,
              method = "rf",
              metric = "ROC",
              tuneGrid = data.frame(mtry = 3),
              ntree = 1000,
              trControl = trainControl(method = "repeatedcv",
                                       repeats = 5,
                                       classProbs = TRUE,
                                       summaryFunction = twoClassSummary))
getTrainPerf(mod0)

roc0 <- roc(testingSet$Class, 
            predict(mod0, testingSet, type = "prob")[,1], 
            levels = rev(levels(testingSet$Class)))
roc0

## Now plot
plot(roc0, print.thres = c(.5), type = "S",
     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
     print.thres.cex = .8, 
     legacy.axes = TRUE)

## Get the model code for the original random forest method:

thresh_code <- getModelInfo("rf", regex = FALSE)[[1]]
thresh_code$type <- c("Classification")
## Add the threshold as another tuning parameter
thresh_code$parameters <- data.frame(parameter = c("mtry", "threshold"),
                                     class = c("numeric", "numeric"),
                                     label = c("#Randomly Selected Predictors",
                                               "Probability Cutoff"))
## The default tuning grid code:
thresh_code$grid <- function(x, y, len = NULL, search = "grid") {
  p <- ncol(x)
  if(search == "grid") {
    grid <- expand.grid(mtry = floor(sqrt(p)), 
                        threshold = seq(.01, .99, length = len))
  } else {
    grid <- expand.grid(mtry = sample(1:p, size = len),
                        threshold = runif(1, 0, size = len))
  }
  grid
}

## Here we fit a single random forest model (with a fixed mtry)
## and loop over the threshold values to get predictions from the same
## randomForest model.
thresh_code$loop = function(grid) {   
  library(plyr)
  loop <- ddply(grid, c("mtry"),
                function(x) c(threshold = max(x$threshold)))
  submodels <- vector(mode = "list", length = nrow(loop))
  for(i in seq(along = loop$threshold)) {
    index <- which(grid$mtry == loop$mtry[i])
    cuts <- grid[index, "threshold"] 
    submodels[[i]] <- data.frame(threshold = cuts[cuts != loop$threshold[i]])
  }    
  list(loop = loop, submodels = submodels)
}

## Fit the model independent of the threshold parameter
thresh_code$fit = function(x, y, wts, param, lev, last, classProbs, ...) { 
  if(length(levels(y)) != 2)
    stop("This works only for 2-class problems")
  randomForest(x, y, mtry = param$mtry, ...)
}

## Now get a probability prediction and use different thresholds to
## get the predicted class
thresh_code$predict = function(modelFit, newdata, submodels = NULL) {
  class1Prob <- predict(modelFit, 
                        newdata, 
                        type = "prob")[, modelFit$obsLevels[1]]
  ## Raise the threshold for class #1 and a higher level of
  ## evidence is needed to call it class 1 so it should 
  ## decrease sensitivity and increase specificity
  out <- ifelse(class1Prob >= modelFit$tuneValue$threshold,
                modelFit$obsLevels[1], 
                modelFit$obsLevels[2])
  if(!is.null(submodels)) {
    tmp2 <- out
    out <- vector(mode = "list", length = length(submodels$threshold))
    out[[1]] <- tmp2
    for(i in seq(along = submodels$threshold)) {
      out[[i+1]] <- ifelse(class1Prob >= submodels$threshold[[i]],
                           modelFit$obsLevels[1], 
                           modelFit$obsLevels[2])
    }
  } 
  out  
}

## The probabilities are always the same but we have to create
## mulitple versions of the probs to evaluate the data across
## thresholds
thresh_code$prob = function(modelFit, newdata, submodels = NULL) {
  out <- as.data.frame(predict(modelFit, newdata, type = "prob"))
  if(!is.null(submodels)) {
    probs <- out
    out <- vector(mode = "list", length = length(submodels$threshold)+1)
    out <- lapply(out, function(x) probs)
  } 
  out 
}



set.seed(949)
mod1 <- train(Class ~ ., data = trainingSet,
              method = thresh_code,
              ## Minimize the distance to the perfect model
              metric = "Dist",
              maximize = FALSE,
              tuneLength = 20,
              ntree = 1000,
              trControl = trainControl(method = "repeatedcv",
                                       repeats = 5,
                                       classProbs = TRUE,
                                       summaryFunction = fourStats))

mod1
library(reshape2)
metrics <- mod1$results[, c(2, 4:6)]
metrics <- melt(metrics, id.vars = "threshold", 
                variable.name = "Resampled",
                value.name = "Data")

ggplot(metrics, aes(x = threshold, y = Data, color = Resampled)) + 
  geom_line() + 
  ylab("") + xlab("Probability Cutoff") +
  theme(legend.position = "top")
