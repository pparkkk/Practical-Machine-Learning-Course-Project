setwd("~/Desktop/R/Statistics_and_Machine_Learning/Practical_Machine_Learning/Practical-Machine-Learning-Course-Project")

#download the file
if (!file.exists("./training.csv")) {
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
              destfile = "./training.csv")
}
if (!file.exists("./validating.csv")) {
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              destfile = "./validating.csv")
}
    
#read into R
training <- read.csv("training.csv")
validating <- read.csv("validating.csv")  #leave the validating set alone for the quiz

names(training)
#select only the columns with measurements relating to belt, forearm, arm and dumbbell
column <- grep(pattern = ".belt.*|.arm.*|.dumbbell.*|classe", x = names(training))
training <- subset(training, select = column)
validating <- subset(validating, select = column)

training[23:25,]
#select only the columns with full information
miss <- c(); none <- c(); nan <- c()
for (i in 1:ncol(training)) {
    col <- training[,i]
    missing <- sum(is.na(col))
    gotinfo <- grep(pattern = ".+", x = col)
    inf <- grep(pattern = "#DIV/0!", x = col)
    if (missing > 0) {miss <- c(miss, i)}
    if (length(gotinfo) != length(col)) {none <- c(none, i)}
    if (length(inf) != 0) {nan <- c(nan,i)}
}

remove <- sort(unique(c(miss, none, nan)))
training <- subset(training, select = -remove)
validating <- subset(validating, select = -remove)

#coerce classe to a factor variable
training$classe <- as.factor(training$classe)
str(training)

#check that there is no near zero variance predictors
library(caret)
nearzero <- nearZeroVar(training[,-53], saveMetrics = TRUE)
identical((nearzero$nzv == FALSE), rep(TRUE, nrow(nearzero)))


#split the training set into train and test sets
set.seed(801)
inTrain <- createDataPartition(y = training$classe, p = 0.75, list = FALSE)
train <- training[inTrain, ] ; test <- training[-inTrain, ]

#exploratory data analysis
corr <- data.frame(variable = names(train), PMCC = rep(NA, ncol(train)))
for (i in 1:ncol(train) - 1) {
    corr[i,2] <- cor(train[,i], as.numeric(train$classe))
}
corr[which.max(abs(corr$PMCC)),]    #only 0.34 -- not a strong linear relationship

corrall <- abs(cor(train[,-53])); diag(corrall) <- 0
highcor <- which(corrall > 0.9, arr.ind = TRUE)   
length(highcor)/2   #number of pairs with correlation higher than 0.9

#use pca to combine highly correlated predictors together
pca <- prcomp(train[,-53])
varex <- pca$sdev^2/sum(pca$sdev^2)
plot(1:52, varex, col = "midnightblue", pch = 16, xlab = "Index", 
     ylab = "Percentage of Variance Explained", 
     main = "Proportion of Variance Explained by Each Measurement")
abline(h = 0.001, col = "orange", lwd = 2)
comp <- length(which(varex >= 0.001))

prep <- preProcess(train[,-53], method = "pca", pcaComp = comp)
trainpca <- predict(prep, train[,-53]); testpca <- predict(prep, test[,-53])
validatingpca <- predict(prep, validating[,-53])
# add the outcome variable into the data frames
trainpca$classe <- train$classe; testpca$classe <- test$classe
validatingpca$classe <- validating$classe


#cross-validation
cvsettings <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

#classification tree
fittree <- train(classe ~ ., data = trainpca, method = "rpart", trControl = cvsettings)
library(rattle)
fancyRpartPlot(fittree$finalModel)

predtree <- predict(fittree, newdata = testpca)
confusionMatrix(predtree, testpca$classe)   #38.36% accurate

#random forest
fitrf <- train(classe ~ ., data = trainpca, method = "rf", trControl = cvsettings)
predrf <- predict(fitrf, newdata = testpca)
confusionMatrix(predrf, testpca$classe) #97.29% accurate

#boosting
fitgbm <- train(classe ~ ., data = trainpca, method = "gbm", trControl = cvsettings, verbose = FALSE)
predgbm <- predict(fitgbm, newdata = testpca)
confusionMatrix(predgbm, testpca$classe)    #81.34% accurate


#on the validating set
predict(fitrf, newdata = validatingpca)

#out of sample
cbind(dim(test)[1], dim(validating)[1])
1 - confusionMatrix(predrf, testpca$classe)$overall['Accuracy']

