# *** CH 05 분석 모형 평가 *** #







##### [1] 회귀 모형 평가 #####





##### <1> 회귀 모형 평가 #####


### (1) RMSE(Root Mean Squared Eror : 평균 제곱근 오차) ###
# rmse(actual, predicted, ...)
# rmse(modelObject)


### (2) MSE(Mean Squared Error : 평균 제곱 오차) ###
# mse(actual, predicted, ...)
# mse(medelObject)


### (3) 결정계수와 조정된 결정계수 ###
# summary(회귀 모형$r.squared)
# summary(회귀 모형$adj.r.squared)


library(ISLR)
hitters <- na.omit(Hitters)
fit_model <- lm(Salary~AtBat+Hits+CWalks+Division+PutOuts,
                data=hitters)
second_model <- lm(Salary~Hits+CWalks+Division+PutOuts,
                   data=hitters)

library(ModelMetrics)
rmse(fit_model)
mse(fit_model)
rmse(second_model)
mse(second_model)

summary(fit_model)$r.squared
summary(fit_model)$adj.r.squared
summary(second_model)$r.squared
summary(second_model)$adj.r.squared





##### <2> 분류 모형 평가 #####


### (1) 혼동 행렬(Confusion Matrix) ###
# 혼동 행렬 : 분석 모델에서 구한 분류의 예측 범주와 데이터의 실제 분류 범주를 교차표(cross Tab) 형태로 정리한 행렬
# confusionMatrix(data, reference, ...) → data : 예측된 분류 데이터 또는 분할표(table 형식), reference : 실제 분류 데이터

library(mlbench)
data("PimaIndiansDiabetes2")

df_pima <- na.omit(PimaIndiansDiabetes2)

set.seed(19190301)
train.idx <- sample(1:nrow(df_pima),
                    size=nrow(df_pima)*0.8)
train <- df_pima[train.idx, ]
test <- df_pima[-train.idx, ]

library(randomForest)
set.seed(19190301)
md.rf <- randomForest(diabetes~.,
                      data=train,
                      ntree=300)
pred <- predict(md.rf, newdata=test)

library(caret)
confusionMatrix(as.factor(pred), test$diabetes)


### (2) AUC(Area Under ROC : AUROC)
# ★ROC 곡선의 X축은 FPR(False Positive Ratio), y축은 TPR(True Positive Ratio)로 두고
#  아랫부분의 면적인 AUC를 기준으로 모형을 평가
# auc(actual, predicted, ...)

library(ModelMetrics)
auc(actual=test$diabetes,
    predicted=as.factor(pred))
# AUC의 값이 0.7과 0.8의 사이(0.7129898)이므로 Fair(보통)의 성능을 보이는 예제
