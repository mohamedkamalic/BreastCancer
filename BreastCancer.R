#Logistic Regression

#Importing dataset
dataset = read.csv("breast cancer.csv" , header = F ,
                   col.names = c("Id Number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"))
dataset = dataset[, 2:11]

dataset$Class <- factor(dataset$Class)
levels(dataset$Class) <- c('benign' ,'malignant')
levels(dataset$Class)

#Taking care of missing data
dataset$Clump.Thickness = ifelse(is.na(dataset$Clump.Thickness),
                                 ave(dataset$Clump.Thickness , FUN = function(x) mean(x , na.rm = T)),
                                 dataset$Clump.Thickness)

dataset$Uniformity.of.Cell.Size = ifelse(is.na(dataset$Uniformity.of.Cell.Size),
                                         ave(dataset$Uniformity.of.Cell.Size , FUN = function(x) mean(x , na.rm = T)),
                                         dataset$Uniformity.of.Cell.Size)

dataset$Uniformity.of.Cell.Shape = ifelse(is.na(dataset$Uniformity.of.Cell.Shape),
                                          ave(dataset$Uniformity.of.Cell.Shape , FUN = function(x) mean(x , na.rm = T)),
                                          dataset$Uniformity.of.Cell.Shape)

dataset$Marginal.Adhesion = ifelse(is.na(dataset$Marginal.Adhesion),
                                   ave(dataset$Marginal.Adhesion , FUN = function(x) mean(x , na.rm = T)),
                                   dataset$Marginal.Adhesion)

dataset$Single.Epithelial.Cell.Size = ifelse(is.na(dataset$Single.Epithelial.Cell.Size),
                                             ave(dataset$Single.Epithelial.Cell.Size , FUN = function(x) mean(x , na.rm = T)),
                                             dataset$Single.Epithelial.Cell.Size)

dataset$Bare.Nuclei = ifelse(is.na(dataset$Bare.Nuclei),
                             ave(dataset$Bare.Nuclei , FUN = function(x) mean(x , na.rm = T)),
                             dataset$Bare.Nuclei)

dataset$Bland.Chromatin = ifelse(is.na(dataset$Bland.Chromatin),
                                 ave(dataset$Bland.Chromatin , FUN = function(x) mean(x , na.rm = T)),
                                 dataset$Bland.Chromatin)

dataset$Normal.Nucleoli = ifelse(is.na(dataset$Normal.Nucleoli),
                                 ave(dataset$Normal.Nucleoli , FUN = function(x) mean(x , na.rm = T)),
                                 dataset$Normal.Nucleoli)

dataset$Mitoses = ifelse(is.na(dataset$Mitoses),
                         ave(dataset$Mitoses , FUN = function(x) mean(x , na.rm = T)),
                         dataset$Mitoses)

#Encoding categorical data

dataset$Class = factor(dataset$Class,
                       levels = c('benign','malignant'),
                       labels = c(0,1) )

#Splitting the dataset into the Training and Test set 

library(caTools)
set.seed(123)
split = sample.split(dataset$Class, SplitRatio = 0.75)
training_set = subset(dataset, split == T)
test_set = subset(dataset, split == F)

#Feature Scaling

training_set[,1:9] =scale(training_set[,1:9])
test_set[,1:9] =scale(test_set[,1:9])


# Fitting Logistic Regression to the Training set

classifier = glm(formula = Class ~ .,
                 family = binomial,
                 data = training_set )

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = training_set[-10])
x_pred = ifelse(prob_pred > 0.5 ,1 , 0)
x_pred

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-10])
y_pred = ifelse(prob_pred > 0.5 ,1 , 0)
y_pred

ogRegcm = table(training_set[,10], x_pred)
LogRegcm
LogRegaccuracy = (LogRegcm[1,1] + LogRegcm[2,2]) / (LogRegcm[1,1] + LogRegcm[2,2] + LogRegcm[1,2] +LogRegcm[2,1])
LogRegaccuracy

# Making the Confusion Matrix 
LogRegcm = table(test_set[,10], y_pred)
LogRegcm
LogRegaccuracy = (LogRegcm[1,1] + LogRegcm[2,2]) / (LogRegcm[1,1] + LogRegcm[2,2] + LogRegcm[1,2] +LogRegcm[2,1])
LogRegaccuracy

#visualising the test set results

#install.packages("ggplot2")            
#install.packages("GGally")
library("ggplot2")                     
library("GGally")                      
ggpairs(test_set[1:5], aes(colour=test_set$Class, alpha=0.4))
ggpairs(test_set[5:9], aes(colour=test_set$Class, alpha=0.4))

# Fitting Kernel SVM to the Training set
#install.packages('e1071')
library(e1071)
classifier = svm(formula = Class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Applying K-Fold Cross Validation
#install.packages('caret')
library(caret)
folds = createFolds(training_set$Class, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  test_fold = training_set[x,]
  classifier = svm(formula = Class ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-10])
  cm = table(test_fold[,10], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] +cm[2,1])
  return(accuracy)
})
cv
K_Fold_accuracy = mean(as.numeric(cv))
K_Fold_accuracy

# Applying Grid Search to find the best parameters
library(caret)
classifier = train(form = Class ~ ., data = training_set , method = 'svmRadial' )
classifier
classifier$bestTune
