setwd("Documents/Kaggle/Titanic - Machine Learning from Disaster/")
library(ggplot2)
rm(list = ls())
train = read.csv("train.csv")
test = read.csv("test.csv")

str(train)

# Identifying those variables that have any NA values or missing values
colSums(is.na(train))
colSums(train=="")


# Replacing the values that have an empty string which can be replaced without much affect
# (i.e., Embarked)
# Through quick inspection, converting the missing values to the most occurring one, S 
table(train$Embarked)
train$Embarked[train$Embarked==""]="S"
train$Embarked = as.factor(train$Embarked)

train$Survived = as.factor(train$Survived)
train$Embarked = droplevels(train$Embarked)

train$Age[is.na(train$Age)] = mean(train$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] = mean(test$Age, na.rm = TRUE)

attach(train)
head(train)

# Relationship between Pclass and survival 
ggplot(data=train, aes(x=Pclass, fill=Survived)) +
  geom_bar(stat="count", position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Pclass")
# 1st class passengers survived on average more than 3rd class 

# Relationship between sex and survival
ggplot(data=train, aes(x=Sex, fill=Survived)) +
  geom_bar(stat="count", position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Sex")
# On average, more female survived

# Relationship between Age and survival 
ggplot(data=train, aes(x=Age, fill=Survived)) +
  geom_histogram(position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Age")
# On average, younger and older people survived 

# Relationship between SibSp and survival
ggplot(data=train, aes(x=SibSp, fill=Survived)) +
  geom_bar() +
  ylab("Frequency") + ggtitle("Survival Rate in terms of SibSp")
# On average, passengers with fewer siblings/spouses survived more often

# Relationship between Parch and survival 
ggplot(data=train, aes(x=Parch, fill=Survived)) +
  geom_bar() +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Parch") 
# On average, passengers with fewer parents/children survived more often

# Relationship between family size and survival
train$FamSize = as.integer(train$SibSp + train$Parch + 1)
ggplot(data=train, aes(x=FamSize, fill=Survived)) +
  geom_bar(stat="count", position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of family size") 
# On average, passengers with a smaller family size survive more often 

# Relationship between Fare and survival
ggplot(data=train, aes(x=Fare, fill=Survived)) +
  geom_histogram(position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Fare") 
# Not conclusive, better to view Pclass instead 

# Relationship between Cabin area and survival
train$cabinID = substr(train$Cabin, 1, 1)
ggplot(train, aes(x=cabinID, fill=Survived)) +
  geom_histogram(stat="count", position="fill")
# Interesting to see that those in cabin B, D & E have a higher chance than A, G or T

# Relationship between port of embarkation and survival 
ggplot(data=train, aes(x=Embarked, fill=Survived)) +
  geom_bar(position="fill") +
  ylab("Frequency") + ggtitle("Survival Rate in terms of port of embarkation") 
# Not conclusive, port of embarkation for your trip would not say much about survival rate 

## _________________________________ Pclass _________________________________
# Relationship between Embarked/Pclass and survival
ggplot(train, aes(x=Embarked, fill=Survived)) +
  geom_histogram(stat="count", position="fill")+
  facet_wrap(~Pclass) +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Embarked and Pclass")
# Also further proves that first class passengers survive on average more than third class passengers

ggplot(train, aes(x=FamSize, fill=Survived)) +
  geom_histogram(stat="count", position="fill")+
  facet_wrap(~Pclass) +
  ylab("Frequency") + ggtitle("Survival Rate in terms of Family size and Pclass")
# Further adding to the point between survival rate and family size
# On average, travelling with a family is safest in first or second class
# In comparison with travelling on third class 

# The title of each passengers name can also be extracted
train$Title <- gsub('(.*, )|(\\..*)', '', train$Name)

train$Title[train$Title == 'Mlle'] = 'Miss' 
train$Title[train$Title == 'Ms'] = 'Miss'
train$Title[train$Title == 'Mme'] = 'Mrs' 
train$Title[train$Title == 'Lady'] = 'Miss'
train$Title[train$Title == 'Dona'] = 'Miss'
Other <- c('Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','the Countess')
train$Title[train$Title %in% Other] = "Other"

train$Title = as.factor(train$Title)

# Perform same function on test data set to be used later on
test$Title <- gsub('(.*, )|(\\..*)', '', test$Name)

test$Title[test$Title == 'Mlle'] = 'Miss' 
test$Title[test$Title == 'Ms'] = 'Miss'
test$Title[test$Title == 'Mme'] = 'Mrs' 
test$Title[test$Title == 'Lady'] = 'Miss'
test$Title[test$Title == 'Dona'] = 'Miss'
Other <- c('Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir','the Countess')
test$Title[test$Title %in% Other] = "Other"

test$Title = as.factor(test$Title)

## _________________________________ Predictions _________________________________
#train = train[c("Pclass", "Sex", "Age", "SibSp", "Parch", "FamSize", "CabinID", "Survived"),]
# Change response variable back to int 
train_comp = train[,c(3,5,6,7,8,10,12,15,2)]
head(train)
train$Embarked = as.factor(train$Embarked)
str(train$Embarked)
# Question 4 ________________
# Decision Tree *****
library(tree)
titanic.tree = tree(Survived ~., data = train_comp)

# Basic idea of model performance (tree)
summary(titanic.tree) 
titanic.tree

plot(titanic.tree)
text(titanic.tree, pretty = 0) # Fix text
title("Predicting survival rate for titanic")

# Naïve Bayes *****
library(e1071)
titanic.bayes = naiveBayes(Survived ~., data = train_comp)

# Bagging *****
library(data.table)
library(caret)
library(adabag)
library(rpart)

titanic.bag = bagging(Survived ~., data = train_comp, mfinal = 5)

# Boosting  *****
titanic.boost = boosting(Survived ~., data = train_comp, mfinal = 5)

# Random Forest  *****
library(randomForest)
titanic.rf = randomForest(Survived ~., data = train_comp, ntree=100)

# Question 5 ________________
# Decision Tree (Testing)
tpredict = predict(titanic.tree, test, type = "class")
test$tree.survived = tpredict

# Naïve Bayes (Testing)
nbpredict = predict(titanic.bayes, test)
test$bayes.survived = nbpredict

# Bagging (Testing)
View(test)
bpredict = predict.bagging(titanic.bag, newdata = test)
test$bag.survived = bpredict$class

table(test$bag.survived)
barplot(WAUSbag$importance[order(WAUSbag$importance, decreasing = TRUE)], ylim = c(0,30), 
        main = "Variable Relative Importance (Bagging)")

# Boosting (Testing)
bopredict = predict.boosting(titanic.boost, newdata = test)
test$boost.survived = bopredict$class

## Overall Feature Importance
rf.importance = as.table(c(titanic.rf$importance[c(3),],0,0,titanic.rf$importance[c(5,1,2,4),],0))
class.var = cbind(titanic.bag$importance, titanic.boost$importance, rf.importance)
class.var = cbind(class.var, rowSums(class.var)) # Sums the row and combines to the dataframe
colnames(class.var) = c("Bagging", "Boosting", "Random Forest", "Total")
class.var = as.data.frame(class.var)
class.var = class.var[order(-class.var$Total),] # sort by descending order (Total)
class.var


#0.77990 decision tree position 13833




