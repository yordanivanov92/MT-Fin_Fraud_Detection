# Using BankSim Data from Edgar Lopez
library(nnet)
library(neuralnet)
library(caTools)
library(caret)
bankSim <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                    header = TRUE,
                    sep = ",")

bankSim_NET <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bsNET140513_032310.csv",
                    header = TRUE,
                    sep = ",")

# Take a smaller random subset for model testing
bankSim_small <- bankSim[sample(nrow(bankSim), 10000), ]

# Keep only relevant columns
bankSim_model <- bankSim_small[, 2:10]
bankSim_model <- bankSim_model[, c("customer", "age", "gender", "merchant", "category", "amount", "fraud")]

# Set seed and split data to train and test subsets
set.seed(1)
split = sample.split(bankSim_model$fraud, SplitRatio = 0.6)

train = subset(bankSim_model, split == TRUE)
test = subset(bankSim_model, split == FALSE)

# Fitting a simple NN with 1 hidden layer
#nn1 <- nnet(fraud~age+gender+category+amount,data = train, size = 10)
#plot(nn1)

# Fitting NN with more hidden layers layers
m <- model.matrix( 
  ~ fraud+age+gender+category+amount, 
  data = train 
)
depvars <- colnames(m)[3:27]
depvars <- gsub("'", "", depvars)
colnames(m) <- c("Intercept","fraud",depvars)
f <- paste(depvars, collapse = ' + ')
f <- paste('fraud ~', f)
f <- as.formula(f)
nn <- neuralnet(f, m, hidden=c(10),linear.output=FALSE)
#plot(nn)

# Fitting the trained NN model to the test dataset
m2 <- model.matrix( 
  ~ fraud+age+gender+category+amount, 
  data = test 
)
depvars_t <- colnames(m2)[3:27]
depvars_t <- gsub("'", "", depvars_t)
colnames(m2) <- c("Intercept","fraud",depvars_t)
res <- compute(nn, m2[, 3:27])
predTest <- res$net.result
predTest <- ifelse(predTest>=0.5,
                   1,
                   0)
# Getting the confusion matrix
predict_table <- table(predTest, m2[,"fraud"])
confusionMatrix(predict_table)
#nn <- neuralnet(f, m, hidden=c(10),linear.output=FALSE)
#predTest    0    1
#       0 3942   18
#       1   13   27

#nn <- neuralnet(f, m, hidden=c(10,10),linear.output=FALSE)
#predTest    0    1
#       0 3936   22
#       1   16   26
# 16 False positives (in reality are not fraud but are classified as fraud by our algorithm)
# 22 False negatives (in reality are fraud but are classified as not fraud by our algorithm)
# 26 fraud case properly classified
# 3936 regular transactions properly classifed

#nn <- neuralnet(f, m, hidden=c(10,10,10),linear.output=FALSE)
#predTest    0    1
#       0 3929   17
#       1   26   28
