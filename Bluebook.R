library(lubridate)
library(mgcv)
library(randomForest)
library(mboost)
library(MASS)

project_dir <- '/Users/ryansteckel/Sandbox/BluebookBull' 

load.train <- function() {
  d <- read.csv(paste(project_dir, '/data/Train.csv', sep=''), header=T)
  months <- month(mdy_hm(d$saledate))
  d$season <- cut(sort(months), breaks=c(0,3,6,9,12), labels=FALSE)
  d
}

load.test <- function() {
  v <- read.csv(paste(project_dir, '/data/Valid.csv', sep=''), header=T)
  months <- month(mdy_hm(v$saledate))
  v$season <- cut(sort(months), breaks=c(0,3,6,9,12), labels=FALSE)
  v
}


d <- load.train()
v <- load.test()


ignored <- c('SalePrice', 'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor', 'fiProductClassDesc', 'state')

factor.names <- names(which(sapply(d, is.factor)))
fs <- factor.names[ which(!factor.names %in% ignored) ]
for(i in 1:length(fs)) {
  all.levels <- union( tolower(levels(d[, fs[i]])), tolower(levels(v[, fs[i]])) )
  d[, fs[i]] <- factor(d[, fs[i]], levels=all.levels)
  v[, fs[i]] <- factor(v[, fs[i]], levels=all.levels)
}

d[d$YearMade == 1000,]$YearMade <- NA
v[v$YearMade == 1000,]$YearMade <- NA

d <- na.roughfix(d)
v <- na.roughfix(v)


#-----EDA-------

str(d)
table(d$YearMade)

#GAM
pg.bl <- d[which(d$ProductGroup == 'bl'),]
pg.mg <- d[which(d$ProductGroup == 'mg'),]
pg.ssl <- d[which(d$ProductGroup == 'ssl'),]
pg.tex <- d[which(d$ProductGroup == 'tex'),]
pg.ttt <- d[which(d$ProductGroup == 'ttt'),]
pg.wl <- d[which(d$ProductGroup == 'wl'),]

table(d$ProductGroup)
boxplot(d$SalePrice ~ d$Enclosure)
boxplot(d$SalePrice ~ d$ProductGroup)
boxplot(d$SalePrice ~ d$season)

boxplot(SalePrice ~ YearMade, data=pg.mg)

boxplot(SalePrice ~ Transmission, data=d)
plot(pg$YearMade, pg$SalePrice)
points(pg$YearMade, predict(fit), col='red')

numeric.columns <- sapply(d, function(c) { 
	is.numeric(c) & length(which(is.na(c))) < 100
})

n <- d[,numeric.columns]

cor(n)


#--------Models-------------

#RF small
fit.rf1 <- randomForest(SalePrice ~ . - saledate - fiModelDesc - fiBaseModel - fiSecondaryDesc - fiModelSeries - fiModelDescriptor - fiProductClassDesc - state, data=d, importance=TRUE, maxnodes=10, ntree=3000, na.action=na.roughfix, do.trace=100)


#RF large
fit.rf2 <- randomForest(SalePrice ~ . - saledate - fiModelDesc - fiBaseModel - fiSecondaryDesc - fiModelSeries - fiModelDescriptor - fiProductClassDesc - state, data=d, importance=TRUE, maxnodes=7, ntree=3000, na.action=na.roughfix, do.trace=100)

varImpPlot(fit.rf1)
print(fit)


fit.bl <- gam(SalePrice ~ s(YearMade), data=pg.bl)
fit.mg <- gam(SalePrice ~ s(YearMade), data=pg.mg)
fit.ssl <- gam(SalePrice ~ s(YearMade), data=pg.ssl)
fit.tex <- gam(SalePrice ~ s(YearMade), data=pg.tex)
fit.ttt <- gam(SalePrice ~ s(YearMade), data=pg.ttt)
fit.wl <- gam(SalePrice ~ s(YearMade), data=pg.wl)

predict.pg <- function(dset) {
	fit <- NULL
	if(dset$ProductGroup == 'BL') { 
		fit <- fit.bl 
	} else if (dset$ProductGroup == 'MG') { 
		fit <- fit.mg 
	} else if (dset$ProductGroup == 'SSL') { 
		fit <- fit.ssl 
	} else if (dset$ProductGroup == 'TEX') { 
		fit <- fit.tex 
	} else if (dset$ProductGroup == 'TTT') { 
		fit <- fit.ttt 
	} else if (dset$ProductGroup == 'WL') { 
		fit <- fit.wl 
	}
	 
	predict(fit, dset, type='response')
}


preds <- matrix(NA, dim(d)[1], 1)
system.time({
for(i in 1:dim(d)[1]) {
	preds[i] <- predict.pg(d[i,])
}})



fit.mboost <- blackboost(d$SalePrice ~ Enclosure + ProductSize + YearMade + ProductGroup, data=d)



#---Train error-----
rmse <- function(obs, est) { sqrt(mean((obs - est)^2)) }

pred.rf1 <- predict(fit.rf1)
pred.rf2 <- predict(fit.rf2)
pred.gam <- preds
pred.mboost <- predict(fit.mboost)


rmse(d$SalePrice, pred.rf1)
rmse(d$SalePrice, pred.rf2)
rmse(d$SalePrice, pred.gam)
rmse(d$SalePrice, pred.mboost)

pred.ens <- rowMeans(cbind(pred.rf1, pred.rf2, pred.gam, pred.mboost))
rmse(d$SalePrice, pred.ens)

fit.ens1 <- lm(d$SalePrice ~ pred.rf1 + pred.rf2 + pred.gam + pred.mboost)
fit.ens1 <- lm(d$SalePrice ~ pred.rf1 + pred.rf2 + pred.mboost)
pred.weighted <- predict(fit.ens1)
rmse(d$SalePrice, pred.weighted)

fit.ens2 <- gam(d$SalePrice ~ s(pred.rf1) + s(pred.rf2) + s(pred.gam) + s(pred.mboost))
fit.ens2 <- gam(d$SalePrice ~ s(pred.rf1) + s(pred.rf2) + s(pred.mboost))
pred.weighted2 <- predict(fit.ens2)
rmse(d$SalePrice, pred.weighted2)

plot(density(d$SalePrice))
lines(density(pred.rf1), col=2)
lines(density(pred.rf2), col=3)
lines(density(pred.gam), col=4)
lines(density(pred.mboost), col=5)
lines(density(pred.weighted2), col=6, lty=2)

#---------Test-------------

tpred.rf1 <- predict(fit.rf1, v)
tpred.rf2 <- predict(fit.rf2, v)

tpred.gam <- matrix(NA, dim(v)[1], 1)
system.time({
for(i in 1:dim(v)[1]) {
	tpred.gam[i] <- predict.pg(v[i,])
}})

tpred.mboost <- predict(fit.mboost, v)

str(fit.ens2$terms)

newd <- data.frame(pred.rf1=tpred.rf1, pred.rf2=tpred.rf2, pred.gam=tpred.gam, pred.mboost=tpred.mboost)
newd <- data.frame(pred.rf1=tpred.rf1, pred.rf2=tpred.rf2, pred.mboost=tpred.mboost)
tpred.submit <- predict(fit.ens2, newd)

tpred.submit[which(tpred.submit < 0)] <- 35000


write.csv(tpred.submit, file=paste(project_dir, '/submissions/submit.csv', sep=''), row.names=FALSE)