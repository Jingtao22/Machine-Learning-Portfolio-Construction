library(readr)
library(mlr)
library(dplyr)
library(randomForest)
library(foreach)
library(Matrix)
library(glmnet)
library(caret)
library(reprtree)



rm(list = ls())
#######simulation code
data<-read.csv('ML_data.csv')
attach(data)
stockname<-names(data)[2:(length(names(data))-2)]
week<-as.character(week)
train<-data[(week>='2010-01')&(week<='2016-52'),]
test<-data[(week>='2017-01'),]
#################data preparation for capm
train_rf<-train[stockname]-train$risk_return
test_rf<-test[stockname]-test$risk_return
train_rf['week']<-train$week
test_rf['week']<-test$week
train_sp<-train$SP_return-train$risk_return
test_sp<-test$SP_return-test$risk_return
###################cap-m
n<-length(train_rf)
my_lms <- lapply(1:(n-1), function(x) lm(train_rf[,x]~train_sp))
coef<-sapply(my_lms, coef)
summaries <- lapply(my_lms, summary)
alphaVector<-coef[1,]   ###########################################################################
betaVector<-coef[2,]
capm.weights<-alphaVector/betaVector
capm.weights<-capm.weights/sum(capm.weights)
pnl<-colSums(capm.weights*t(as.matrix(test[stockname])))
#pnl[c(TRUE, FALSE)]
capm.pnl<-data.frame(matrix(ncol = 1, nrow = 52))
colnames(capm.pnl)<-c('pnl')
capm.pnl$pnl<- exp(cumsum(pnl))
plot(capm.pnl$pnl, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="capm test pnl")
sharpe<-(capm.pnl$pnl[52]-1)/sd(capm.pnl$pnl)
##############################farm french
data.farma<-read.csv('F-F_Research_Data_Factors_weekly-2.csv')
attach(data.farma)
train.farma<-data.farma[(X>='201001')&(X<'201701'),]
test.farma<-data.farma[(X>='201701')&(X<'201801'),]
ewma<-read.csv('3weekmavg.csv')
train.ewma<-ewma[c(2:366),paste(stockname, "3weekmavg",sep='')]
test.ewma<-ewma[c(367:418),paste(stockname, "3weekmavg",sep='')]

###################after data prepration, we train our model and get predicted return
##################on test set for every week
farma_lms<-lapply(1:(n-1), function(x) lm(train_rf[,x]~train_sp+train.farma$HML+train.farma$SMB+train.ewma[,x]))
coef<-sapply(farma_lms, coef)
prediction<-data.frame(matrix(ncol = 498, nrow = 52))
hit.rate.farma<-data.frame(matrix(nrow=498,ncol=1))
colnames(hit.rate.farma)<-c('hit_rate')
for(i in 1:498){
  prediction[,i] <- coef[1,i]+coef[2,i]*test_sp+coef[3,i]*test.farma$HML+coef[4,i]*test.farma$SMB+coef[5,i]*test.ewma[,i]
  hit.rate.farma[i,'hit_rate']<-sum(abs(sign(prediction[,i])+sign(test[,stockname[i]])))/2/52
}
colnames(prediction)<-stockname
mean(as.matrix(hit.rate.farma))
##################we construct our portforlio for four different ways
##########first buy every odd weeks with equal weights
#######we will buy if predict >0 and sell if <0 
farma.pnl<-data.frame(matrix(ncol = 4, nrow = 52))
sharpe<-data.frame(matrix(ncol = 1, nrow = 4))
colnames(sharpe)<-c('sharpe_ratio')
colnames(farma.pnl)<-c('pnl_oddweek','pnl_evenweek','pnl_capmweightsodd','pnl_capmweightseven')
for (i in seq(52)){
  if (i%%2 !=0){
    weights<-ifelse(prediction[i,]>=0, 1,-1)
    if (sum(weights)!=0){                       ####分母！=0
        weights<-weights/abs(sum(weights))}
  farma.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
  else{
  weights<-rep(0, 498)    ##########################################################################0多次
  farma.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
}

farma.pnl$pnl_oddweek<- exp(cumsum(farma.pnl$pnl_oddweek))
plot(farma.pnl$pnl_oddweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="farma test pnl")
sharpe[1,'sharp_ratio']<-(farma.pnl$pnl_oddweek[52]-1)/sd(farma.pnl$pnl_oddweek)
#######################then we try to buy every even weeks
for (i in seq(52)){
  if (i%%2 ==0){
      weights<-ifelse(prediction[i,]>=0, 1, -1)
      if (sum(weights)!=0){weights<-weights/abs(sum(weights))}
      farma.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    farma.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
}
farma.pnl$pnl_evenweek<- exp(cumsum(farma.pnl$pnl_evenweek))
plot(farma.pnl$pnl_evenweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="farma test pnl")
sharpe[2,'sharpe_ratio']<-(farma.pnl$pnl_evenweek[52]-1)/sd(farma.pnl$pnl_evenweek)
########################## then try to go with capm weights
for (i in seq(52)){
  if (i%%2 !=0){
    weights<-ifelse(prediction[i,]>=0, 1, -1)
    weights<-capm.weights*weights
    if (sum(weights)!=0){
      weights<-weights/sum(weights)}
    farma.pnl[i,'pnl_capmweightsodd']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    farma.pnl[i,'pnl_capmweightsodd']<-sum(weights*test[i,stockname])
  }
}
farma.pnl$pnl_capmweightsodd<- exp(cumsum(farma.pnl$pnl_capmweightsodd))
plot(farma.pnl$pnl_capmweightsodd, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="farma test pnl")
sharpe[3,'sharpe_ratio']<-(farma.pnl$pnl_capmweightsodd[52]-1)/sd(farma.pnl$pnl_capmweightsodd)
##############lastly
for (i in seq(52)){
  if (i%%2 ==0){
    weights<-ifelse(prediction[i,]>=0, 1, -1)
    weights<-capm.weights*weights
    if (sum(weights)!=0){
      weights<-weights/abs(sum(weights))}
    farma.pnl[i,'pnl_capmweightseven']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    farma.pnl[i,'pnl_capmweightseven']<-sum(weights*test[i,stockname])
  }
}
farma.pnl$pnl_capmweightseven<- exp(cumsum(farma.pnl$pnl_capmweightseven))
plot(farma.pnl$pnl_capmweightseven, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="farma test pnl")
sharpe[4,'sharpe_ratio']<-(farma.pnl$pnl_capmweightseven[52]-1)/sd(farma.pnl$pnl_capmweightseven)







################################apt with several different factors
data.apt<-read.csv('Final merge result.csv')
attach(data.apt)
train.apt<-data.apt[(week>='2010-01')&(week<'2017-01'),]
test.apt<-data.apt[(week>='2017-01')&(week<'2018-01'),]

###################first we apply random foerest
##################create two temptational dataframe to train the stock one
i=1
train.apt$return<-train[,i+1]
train.apt$mavg<-train.ewma[,i]
test.apt$return<-test[,i+1]
test.apt$mavg<-test.ewma[,i]
######train for stock one to find the best ntree,mtry and maxnodes parameters
set.seed(1234)
trControl <- trainControl(method = "cv",number = 3,search = "grid")
rf_default <- train(return~copper_futures_return+crude_oil_return+gol_.futures_return+
                      Euro_Dollar._return+Nasdaq_return+Yuan_Dollar_return+
                      X1._year_Treasury_Rate_Yield_return+X10_year_Treasury_Bond_Rate_Yield_Return
                    +mavg,
                    data = train.apt,
                    method = "rf",
                    trControl = trControl)
print(rf_default)
best_mtry<-rf_default$bestTune$mtry
####max nodes
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(return~copper_futures_return+crude_oil_return+gol_.futures_return+
                        Euro_Dollar._return+Nasdaq_return+Yuan_Dollar_return+
                        X1._year_Treasury_Rate_Yield_return+X10_year_Treasury_Bond_Rate_Yield_Return
                      +mavg,
                      data = train.apt,
                      method = "rf",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
max_nodes<-7############by observe the summary 7 is the best maxnode
####ntree
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(return~copper_futures_return+crude_oil_return+gol_.futures_return+
                         Euro_Dollar._return+Nasdaq_return+Yuan_Dollar_return+
                         X1._year_Treasury_Rate_Yield_return+X10_year_Treasury_Bond_Rate_Yield_Return
                       +mavg,
                       data = train.apt,
                       method = "rf",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes =9,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)
best_ntree<-800 #observe the summary800 is the best


fit_rf<-randomForest(return~copper_futures_return+crude_oil_return+gol_.futures_return+
               Euro_Dollar._return+Nasdaq_return+Yuan_Dollar_return+
               X1._year_Treasury_Rate_Yield_return+X10_year_Treasury_Bond_Rate_Yield_Return
             +mavg,data=train.apt,ntree=best_ntree, mtry=best_mtry, maxnodes = max_nodes)

prediction <-predict(fit_rf, test.apt)
hit.rate<-sum(abs(sign(prediction)+sign(test[,i+1])))/2/52# predict sign accuracy
hit.rate
#################plot tree
reprtree:::plot.getTree(fit_rf)

#############now we get into loop
prediction<-data.frame(matrix(nrow=52,ncol=498))
colnames(prediction)<-stockname
hit.rate<-data.frame(matrix(nrow=498,ncol=1))
colnames(hit.rate)<-c('hit_rate')
for (i in 1:498){
  print(stockname[i])
  train.apt$return<-train[,i+1]
  train.apt$mavg<-train.ewma[,i]
  test.apt$return<-test[,i+1]
  test.apt$mavg<-test.ewma[,i]
  fit_rf<-randomForest(return~copper_futures_return+crude_oil_return+gol_.futures_return+
                         Euro_Dollar._return+Nasdaq_return+Yuan_Dollar_return+
                         X1._year_Treasury_Rate_Yield_return+X10_year_Treasury_Bond_Rate_Yield_Return
                       +mavg,data=train.apt,ntree=best_ntree, mtry=best_mtry, maxnodes = max_nodes)

  prediction[,stockname[i]] <-predict(fit_rf, test.apt)
  hit.rate[i,'hit_rate']<-sum(abs(sign(prediction[,stockname[i]])+sign(test.apt$return)))/2/52# predict sign accuracy
  print('finish')
}
mean(as.matrix(hit.rate))
##################we construct our portforlio for two different ways
##########first buy every odd weeks with equal weights
#######we will buy if predict >0 and sell if <0 
RF.pnl<-data.frame(matrix(ncol = 2, nrow = 52))
sharpe2<-data.frame(matrix(ncol = 1, nrow = 2))
colnames(sharpe2)<-c('sharpe_ratio')
colnames(RF.pnl)<-c('pnl_oddweek','pnl_evenweek')
for (i in seq(52)){
  if (i%%2 !=0){
    weights<-ifelse(prediction[i,]>=0, 1,-1)
    if (sum(weights)!=0){
      weights<-weights/abs(sum(weights))}
    RF.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    RF.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
}

RF.pnl$pnl_oddweek<- exp(cumsum(RF.pnl$pnl_oddweek))
plot(RF.pnl$pnl_oddweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="APT test pnl(odd number of weeks)")
sharpe2[1,'sharpe_ratio']<-(RF.pnl$pnl_oddweek[52]-1)/sd(RF.pnl$pnl_oddweek)
#######################then we try to buy every even weeks
for (i in seq(52)){
  if (i%%2 ==0){
    weights<-ifelse(prediction[i,]>=0, 1, -1)
    if (sum(weights)!=0){weights<-weights/abs(sum(weights))}
    RF.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    RF.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
}
RF.pnl$pnl_evenweek<- exp(cumsum(RF.pnl$pnl_evenweek))
plot(RF.pnl$pnl_evenweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="APT test pnl(even number of weeks)")
sharpe2[2,'sharpe_ratio']<-(RF.pnl$pnl_evenweek[52]-1)/sd(RF.pnl$pnl_evenweek)









##########################last but not least, we do elastic net 多项式 lasso and ridge 
colnames(data.apt)<-c('week','X1','X2','X3','X4','X5','X6','X7','X8')
data.EN<-data.apt
newdata<-log(data.apt[c('X1','X2','X3','X4','X5','X6','X7','X8')]+1)
names(newdata) <- paste0(c('X1','X2','X3','X4','X5','X6','X7','X8'), "_log")
newdata$week<-data.EN$week
data.EN<-merge(data.EN,newdata,by='week',all = TRUE)
newdata<-exp(data.apt[c('X1','X2','X3','X4','X5','X6','X7','X8')]+1)
names(newdata) <- paste0(c('X1','X2','X3','X4','X5','X6','X7','X8'), "_exp")
newdata$week<-data.EN$week
data.EN<-merge(data.EN,newdata,by='week',all = TRUE)
newdata<-data.apt[c('X1','X2','X3','X4','X5','X6','X7','X8')]^2
names(newdata) <- paste0(c('X1','X2','X3','X4','X5','X6','X7','X8'), "^2")
newdata$week<-data.EN$week
data.EN<-merge(data.EN,newdata,by='week',all = TRUE)
newdata<-data.apt[c('X1','X2','X3','X4','X5','X6','X7','X8')]^3
names(newdata) <- paste0(c('X1','X2','X3','X4','X5','X6','X7','X8'), "^3")
newdata$week<-data.EN$week
data.EN<-merge(data.EN,newdata,by='week',all = TRUE)
train.EN<-data.EN[(week>='2010-01')&(week<'2017-01'),]
train.EN<-train.EN[,colnames(train.EN)[2:41]]
test.EN<-data.EN[(week>='2017-01')&(week<'2018-01'),]
test.EN<-test.EN[,colnames(test.EN)[2:41]]
#############now we get into loop
prediction2<-data.frame(matrix(nrow=52,ncol=498))
colnames(prediction2)<-stockname
hit.rate2<-data.frame(matrix(nrow=498,ncol=1))
colnames(hit.rate2)<-c('hit_rate')
for (i in 1:498){
  print(stockname[i])
  set.seed(1234)
  train.EN$mavg<-train.ewma[,i]
  train.EN$return<-train[,i+1]
  test.EN$mavg<-test.ewma[,i]
  test.EN$return<-test[,i+1]
  x <- model.matrix(~.,data = train.EN[,1:41])
  y <- as.matrix(train.EN[,42])
  fit.en<-cv.glmnet(x,y)
  prediction2[,stockname[i]] <-predict(fit.en,model.matrix(~.,data = test.EN[,1:41]))
  hit.rate2[i,'hit_rate']<-sum(abs(sign(prediction2[,stockname[i]])+sign(test.EN$return)))/2/52# predict sign accuracy
  print('finish')
}
mean(as.matrix(hit.rate2))
##################we construct our portforlio for two different ways
##########first buy every odd weeks with equal weights
#######we will buy if predict >0 and sell if <0 
EN.pnl<-data.frame(matrix(ncol = 2, nrow = 52))
sharpe3<-data.frame(matrix(ncol = 1, nrow = 2))
colnames(sharpe3)<-c('sharpe_ratio')
colnames(EN.pnl)<-c('pnl_oddweek','pnl_evenweek')
for (i in seq(52)){
  if (i%%2 !=0){
    weights<-ifelse(prediction2[i,]>=0, 1,-1)
    if (sum(weights)!=0){
      weights<-weights/abs(sum(weights))}
    EN.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    EN.pnl[i,'pnl_oddweek']<-sum(weights*test[i,stockname])
  }
}

EN.pnl$pnl_oddweek<- exp(cumsum(EN.pnl$pnl_oddweek))
plot(EN.pnl$pnl_oddweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="EN test pnl")
sharpe3[1,'sharpe_ratio']<-(EN.pnl$pnl_oddweek[52]-1)/sd(EN.pnl$pnl_oddweek)
#######################then we try to buy every even weeks
for (i in seq(52)){
  if (i%%2 ==0){
    weights<-ifelse(prediction2[i,]>=0, 1, -1)
    if (sum(weights)!=0){weights<-weights/abs(sum(weights))}
    EN.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
  else{
    weights<-rep(0, 498)
    EN.pnl[i,'pnl_evenweek']<-sum(weights*test[i,stockname])
  }
}
EN.pnl$pnl_evenweek<- exp(cumsum(EN.pnl$pnl_evenweek))
plot(EN.pnl$pnl_evenweek, type="l", col="red", lwd=5, xlab="weeks", ylab="pnl", main="EN test pnl")
sharpe3[2,'sharpe_ratio']<-(EN.pnl$pnl_evenweek[52]-1)/sd(EN.pnl$pnl_evenweek)













