setwd("C:/Users/ragha/Desktop/Raghav/Great Learning/Projects/Project 4")
library(readxl)
library(corrplot)
library(class)
library(gridExtra)
library(tidyverse)
library(car)
library(class)
library(caret)
library(ROCR)
library(ineq)
library(dplyr)
library(broom)
library(caTools)
churn <- read_excel("Cellphone-1.xlsx", sheet = 2)
churn <- as.data.frame(churn)

head(churn)
tail(churn)
str(churn)
summary(churn)


prop.table(table(churn$Churn))*100


#Histogram
par(mfrow = c(3,3))
for(i in names(churn[-c(1,3,4)])){
  hist(churn[,i], xlab = names(churn[i]), col = "red", border = "black", ylab = "Frequency",
       main =paste("Histogram of", names(churn[i])),col.main = "darkGreen")
}
par(mfrow = c(1,1))

#Barplot
table1 <- churn %>% group_by(Churn,ContractRenewal) %>% summarise("Values" = n())
table1[c(1,2)] <- lapply(table1[c(1,2)], as.factor)
A <- ggplot(table1,aes(ContractRenewal,Values,fill = Churn)) + geom_bar(stat = "identity",color = "Black", position = 'dodge')+
  geom_text(aes(label = Values), 
            size = 3, 
            color = "black",
            position = position_dodge(width = 0.9),
            vjust = -0.2)


table2 <- churn %>% group_by(Churn,DataPlan) %>% summarise("Values" = n())
table2[c(1,2)] <- lapply(table2[c(1,2)], as.factor)
B <- ggplot(data = table2, aes(DataPlan,Values, fill = Churn)) + geom_bar(stat = 'identity', position = 'dodge',color = "Black")+
  geom_text(aes(label = Values),
            size = 3,
            vjust = -0.2,
            position = position_dodge(width = 0.9))

grid.arrange(A,B)

#Boxplot
A <- ggplot(data = churn, aes(as.factor(Churn),AccountWeeks)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("No. of active account weeks vs Churn")
B <- ggplot(data = churn, aes(as.factor(Churn),DataUsage)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Data Usage vs Churn")
C <- ggplot(data = churn, aes(as.factor(Churn),CustServCalls)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Customer Service Calls vs Churn")
D <- ggplot(data = churn, aes(as.factor(Churn),DayMins)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Average daytime minutes vs Churn")
E <- ggplot(data = churn, aes(as.factor(Churn),DayCalls)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Average daytime calls vs Churn")
G <- ggplot(data = churn, aes(as.factor(Churn),MonthlyCharge)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Monthly Charge vs Churn")
H <- ggplot(data = churn, aes(as.factor(Churn),OverageFee)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Overage fee vs Churn")
I <- ggplot(data = churn, aes(as.factor(Churn),RoamMins)) + geom_boxplot(color = c("Red","Blue")) + xlab("Churn") + ggtitle("Roaming Minutes vs Churn")

grid.arrange(A,B,C,D,E,G,H,I)
par(mfrow =c(1,1))

#Missing values
any(is.na(churn))


#Outlier Detection and Treatment 

#Cooks distance for outliers detection

cooksd <- cooks.distance(glm(Churn~.,data = churn))
plot(cooksd,pch="*", cex =1, main ="Cook's Distance", col ="blue")
abline(h = 4*mean(cooksd, na.rm = TRUE), col = "DarkRED",lwd = 2)
#Any value above red line  indicates influential values or outliers

#Treatment using mean imputation method
mydata <- churn[,c(2,5,7:11)]

Outliers<- mydata[1:50,]


for(i in c(1:7)) {
  Outliers[,i] <- NA
  Box <-boxplot(mydata[,i],plot =F)$out
  if (length(Box)>0){
    Outliers[1:length(Box),i] <- Box
  }
  mydata[which(mydata[,i] %in% Outliers[,i]),i] <- round(mean(mydata[-which(mydata[,i] %in% Outliers[,i]),i]))
  
}

mydata <- cbind.data.frame(churn[,c(1,3,4,6)],mydata)

churn <- mydata


#Multicollinearity
cor(churn)
corrplot.mixed(cor(churn),lower = "number", upper = "pie")

#Variance Inflation Factor
vif(glm(Churn~., data = churn))
vif(glm(Churn~.-DataUsage, data = churn))
vif(glm(Churn~.-DataUsage - MonthlyCharge, data = churn))

#DataUsage and MonthlyCharge are removed from the  models


#-------------------------------------------------------------------
#1. Logistic Regression

#Testing assumptions of logistic regression 

#1. Linear relationship b/w log odds and independent variables
model <- glm(Churn~., data = churn, family = binomial)
prob <- predict(model, type = "response", churn)
mydata <- churn
mydata[,c(1,2,3)] <- lapply(mydata[,c(1,2,3)], as.factor)


mydata<- mydata %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)

# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(prob/(1-prob))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

#2.Outliers
#3. Multicollinearity


#Model Formation

log.churn <- churn[-c(6,9)]

library(caTools)
set.seed(100)
split <- sample.split(log.churn$Churn,SplitRatio = 2/3)
log.train <- log.churn[split == TRUE,]
log.test <- log.churn[split == FALSE,]

#Feature Scaling
log.train[-1] <- scale(log.train[-1])

log.test[-1] <- scale(log.test[-1])


#Stepwise logistic regression with 10 - fold cross validation
train.control <- trainControl(method = "cv", number = 10)
model <- train(as.factor(Churn)~., data = log.train, method = "glm",trControl = train.control,family =binomial())
summary(model)


model <- train(as.factor(Churn)~. - AccountWeeks, data = log.train, method = "glm",trControl = train.control,family =binomial(link = "logit"))
summary(model)


plot(varImp(model), main = "Important Variables")


log.train$prob.pred <- predict(model,newdata = log.train, type = "prob")[,"1"]
log.train$pred.class <- predict(model,log.train, type = "raw")

log.cm.train <-confusionMatrix(as.factor(log.train$Churn),as.factor(log.train$pred.class))
log.cm.train

#Predicting Testing data set

log.test$prob.pred <- predict(model,newdata = log.test, type = "prob")[,"1"]
log.test$pred.class <- predict(model,log.test, type = "raw")

log.cm.test <-confusionMatrix(as.factor(log.test$Churn),as.factor(log.test$pred.class))
log.cm.test


log.test.Accuracy <- log.cm.test$overall[[1]]
log.test.Accuracy

log.test.class.err <- 1 - log.test.Accuracy
log.test.class.err

log.test.Sensitivity <- log.cm.test$byClass[[1]] 
log.test.Sensitivity

log.test.Specificity <- log.cm.test$byClass[[2]]
log.test.Specificity


#ROCR
log.test.pred <- prediction(log.test$Churn,log.test$pred.class)
log.test.perf <- performance(log.test.pred,"tpr","fpr")
plot(log.test.perf, col = "Red",lwd =2, main = "ROC in Logistic regression")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.77,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
log.test.auc <- performance(log.test.pred,"auc")
log.test.auc <- log.test.auc@y.values[[1]]
log.test.auc <- log.test.auc[[1]]
log.test.auc


#KS
log.test.ks <- log.test.perf@y.values[[1]] - log.test.perf@x.values[[1]]
log.test.ks <- log.test.ks[2]
log.test.ks

#Gini
log.test.gini <- ineq(log.test$prob.pred,type ="Gini")
log.test.gini

#Concordance
library(InformationValue)
log.test.cord <- Concordance(actuals = log.test$Churn,predictedScores = log.test$prob.pred)
log.test.cord<- log.test.cord$Concordance
log.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.log <- t(data.frame(log.test.Accuracy,log.test.class.err,log.test.Sensitivity,log.test.Specificity,log.test.ks,log.test.auc,log.test.gini,log.test.cord))
row.names(test.model.log) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
test.model.log


#-------------------------------------------------------------------------------
# KNN

knn.churn <- churn

#Splitting of dataset
set.seed(100)
split <- sample.split(knn.churn$Churn,SplitRatio = 2/3)
knn.train <- knn.churn[split == TRUE,]
knn.test <- knn.churn[split == FALSE,]

#Feature Scaling
knn.train[-1] <- scale(knn.train[-1])

knn.test[-1] <- scale(knn.test[-1])

#Model Formation
control <- trainControl(method = 'cv', number = 10)

kn <- train(as.factor(Churn) ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:20),
             trControl  = control,
             metric     = "Accuracy",
             data       = knn.train)
kn
plot(kn,main= "KNN")
plot(varImp(kn),main = "Important Variables")

knn.train$pred.class <- predict(kn,newdata = knn.train,type = "raw")
knn.train$pred.prob <- predict(kn, newdata = knn.train, type ="prob")[,"1"]
confusionMatrix(as.factor(knn.train$Churn),knn.train$pred.class)

#Testing on 

knn.test$pred.class <- predict(kn,newdata = knn.test, type = "raw")
knn.test$pred.prob <-  predict(kn, newdata = knn.test, type = "prob")[,"1"]

knn.cm.test <-confusionMatrix(as.factor(knn.test$Churn),knn.test$pred.class)
knn.cm.test

knn.test.Accuracy <- knn.cm.test$overall[[1]]
knn.test.Accuracy

knn.test.class.err <- 1 - knn.test.Accuracy
knn.test.class.err

knn.test.Sensitivity <- knn.cm.test$byClass[[1]] 
knn.test.Sensitivity

knn.test.Specificity <- knn.cm.test$byClass[[2]]
knn.test.Specificity


#ROCR
knn.test.pred <- prediction(knn.test$Churn,knn.test$pred.class)
knn.test.perf <- performance(knn.test.pred,"tpr","fpr")
plot(knn.test.perf, col = "Red", lwd =2, main ="ROC in KNN")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

#AUC
knn.test.auc <- performance(knn.test.pred,"auc")
knn.test.auc <- knn.test.auc@y.values[[1]]
knn.test.auc <- knn.test.auc[[1]]
knn.test.auc

#KS
knn.test.ks <- knn.test.perf@y.values[[1]] - knn.test.perf@x.values[[1]]
knn.test.ks <- knn.test.ks[2]
knn.test.ks

#Gini
knn.test.gini <- ineq(knn.test$pred.prob,type ="Gini")
knn.test.gini

#Concordance
library(InformationValue)
knn.test.cord <- Concordance(actuals = knn.test$Churn,predictedScores = knn.test$pred.prob)
knn.test.cord<- knn.test.cord$Concordance
knn.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.knn <- t(data.frame(knn.test.Accuracy,knn.test.class.err,knn.test.Sensitivity,knn.test.Specificity,knn.test.ks,knn.test.auc,knn.test.gini,knn.test.cord))
row.names(test.model.knn) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
test.model.knn


#-------------------------------------------------------------------------------
#Naive Bayes


nb.churn <- churn

#Splitting of dataset
set.seed(100)
split <- sample.split(nb.churn$Churn,SplitRatio = 2/3)
nb.train <- nb.churn[split == TRUE,]
nb.test <- nb.churn[split == FALSE,]

#Model Formation
control <- trainControl(method = 'cv', number = 10)

search_grid <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = 0:5,
  adjust = seq(0, 5, by = 1)
)

nb <- train(
  x = nb.train[-1],
  y = as.factor(nb.train$Churn),
  method = "nb",
  trControl = control,
  tuneGrid = search_grid
)

nb
nb$results %>% top_n(5, wt = Accuracy) %>% arrange(desc(Accuracy))

plot(nb)
plot(varImp(nb), main = "Important Variable")

nb.train$pred.class <- predict(nb,newdata = nb.train,type = "raw")
nb.train$pred.prob <- predict(nb, newdata = nb.train, type ="prob")[,"1"]
confusionMatrix(as.factor(nb.train$Churn),nb.train$pred.class)

#Testing of model

nb.test$pred.class <- predict(nb,newdata = nb.test, type = "raw")
nb.test$pred.prob <-  predict(nb, newdata = nb.test, type = "prob")[,"1"]

nb.cm.test <-confusionMatrix(as.factor(nb.test$Churn),nb.test$pred.class)
nb.cm.test

nb.test.Accuracy <- nb.cm.test$overall[[1]]
nb.test.Accuracy

nb.test.class.err <- 1 - nb.test.Accuracy
nb.test.class.err

nb.test.Sensitivity <- nb.cm.test$byClass[[1]] 
nb.test.Sensitivity

nb.test.Specificity <- nb.cm.test$byClass[[2]]
nb.test.Specificity


#ROCR
nb.test.pred <- prediction(nb.test$Churn,nb.test$pred.class)
nb.test.perf <- performance(nb.test.pred,"tpr","fpr")
plot(nb.test.perf, col = "Red", lwd = 2, main ="ROC in Naive Bayes")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,0.8))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


#AUC
nb.test.auc <- performance(nb.test.pred,"auc")
nb.test.auc <- nb.test.auc@y.values[[1]]
nb.test.auc <- nb.test.auc[[1]]
nb.test.auc

#KS
nb.test.ks <- nb.test.perf@y.values[[1]] - nb.test.perf@x.values[[1]]
nb.test.ks <- nb.test.ks[2]
nb.test.ks

#Gini
nb.test.gini <- ineq(nb.test$pred.prob,type ="Gini")
nb.test.gini

#Concordance
library(InformationValue)
nb.test.cord <- Concordance(actuals = nb.test$Churn,predictedScores = nb.test$pred.prob)
nb.test.cord<- nb.test.cord$Concordance
nb.test.cord
detach("package:InformationValue", unload = TRUE)


test.model.nb <- t(data.frame(nb.test.Accuracy,nb.test.class.err,nb.test.Sensitivity,nb.test.Specificity,nb.test.ks,nb.test.auc,nb.test.gini,nb.test.cord))
row.names(test.model.nb) <- c("Accuracy","Classification error","Sensitivity","Specificity","KS stat", "AUC", "Gini", "Concordance")
test.model.nb

#Combining performance measures of all models
combined.model <- cbind(test.model.knn,test.model.log,test.model.nb)
colnames(combined.model) <- c("KNN","Logistic","Naive Bayes")
combined.model

#ROC curve of all models
par(mfrow = c(2,2))

plot(log.test.perf, col = "Red",lwd =2, main = "ROC in Logistic regression")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.77,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


plot(knn.test.perf, col = "Red", lwd =2, main ="ROC in KNN")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)


plot(nb.test.perf, col = "Red", lwd = 2, main ="ROC in Naive Bayes")
abline(a = 0.0, b = 1.0, col = "Blue", lwd = 2)
abline(h =1,v = 0,col =  "Purple", lwd = 2)
text(x = c(0.8,0.4,0.1), y = c(0.2,0.6,0.9), labels = c("A(0.5)","B","C"), cex = 1.5)
text(x = c(0.8,0.5,0.3),y =c(0.6,0.85,0.96), labels = c("Random model line","ROC","Perfect Model"),col = c("Blue","Red","Purple"), cex = c(1.2,1.2,1.2))
arrows(x0 = 0.8,y0 = 0.77,x1 = 0.8, y1 = 0.65,col = "Blue", lwd = 2)

par(mfrow = c(1,1))

