# Using BankSim Data from Edgar Lopez
library(nnet)
library(neuralnet)
library(caTools)
library(caret)
#library(Rmpi)
bankSim <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bs140513_032310.csv",
                    header = TRUE,
                    sep = ",")

# bankSim_NET <- read.csv(file = "C:/Users/Yordan Ivanov/Desktop/Master Thesis Project/data/bank_sim_synthetic/bsNET140513_032310.csv",
#                     header = TRUE,
#                     sep = ",")

# Take a smaller random subset for model testing
set.seed(1)
bankSim_small <- bankSim[sample(nrow(bankSim), 10000), ]

# Keep only relevant columns
bankSim_model <- bankSim_small[, 2:10]
bankSim_model <- bankSim_model[, c("customer", "age", "gender", "merchant", "category", "amount", "fraud")]

# Set seed and split data to train and test subsets

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
nn <- neuralnet(f, m, hidden=c(10), linear.output=FALSE)
nn_crossentropy <- neuralnet(f, m, hidden=c(10), linear.output=FALSE, err.fct = "ce")
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
res_crossentropy <- compute(nn_crossentropy, m2[, 3:27])
predTest <- res$net.result
predTest_crossentropy <- res_crossentropy$net.result
predTest <- ifelse(predTest>=0.5,
                   1,
                   0)
predTest_crossentropy <- ifelse(predTest_crossentropy>=0.5,
                                1,
                                0)
# Getting the confusion matrix
predict_table <- table(predTest, m2[,"fraud"])
confusionMatrix(predict_table)

predict_table_crossentropy <- table(predTest_crossentropy, m2[, "fraud"])
confusionMatrix(predict_table_crossentropy)

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

#nproc = 2 # could be automatically determined
# Specify one master and nproc-1 slaves
# Rmpi:: mpi.spawn.Rslaves(nslaves=2)
# # Execute function "func_to_be_parallelized" on multiple CPUs; pass two variables to function
# my_fast_results = Rmpi::mpi.parLapply(f,
#                                       neuralnet,
#                                       m,
#                                       c(10),
#                                       FALSE)
# # Close slaves
# Rmpi::mpi.close.Rslaves(dellog=T)

detach(package:neuralnet, unload = TRUE)
library(ROCR)
predict.testing <- prediction(predTest, m2[,"fraud"])
pref <- performance(predict.testing, "tpr", "fpr")
plot(pref)
predict.testing
