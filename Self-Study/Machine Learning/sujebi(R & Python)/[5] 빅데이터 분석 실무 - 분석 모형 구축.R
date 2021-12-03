# *** CH 04 분석 모형 구축 *** #







##### [1] 통계 분석 #####





##### <1> 주성분 분석(PCA) #####


### (1) 주성분 분석(PCA : Principal Component Analysis)의 개념 ###

# 주성분 분석 : 서로 상관성이 높은 변수를 상관관계가 없는 소수의 변수로 변환하는 '차원 축소 기법'
# ★상관관계가 있는 고차원 자료를 자료의 변동을 최대한 보존하는 저차원 자료로 변환하는 차원 축소 방법


### (2) 주성분의 선택 방법 ###

# 누적 기여율 : 누적 기여율이 85% 이상인 지점까지를 주성분의 수로 결정
# 스크리 산점도(Scree Plot) : 스크리 산점도의 기울기가 완만해지기 ★직전★까지의 주성분 수로 결정


### (3) 주성분 분석 구축 프로세스 ###

# ① 주성분 분석 함수 #

# princomp(x, cor, scores, ...) → cor : 공분산 행렬 또는 상관 행렬(default) 사용 여부, scores : 각 주성분의 점수 계산 여부
# 고유벡터(eigenvaetors)가 loadings 변수에 저장
# 변수들의 선형 결합을 통해 변환된 값을 주성분 점수라고 하고, scores 변수를 통해 확인 가능
iris_pca <- princomp(iris[ , -5],   # Species 제외
                     cor=FALSE,
                     scores=TRUE)

summary(iris_pca)   # 누적 기여율

plot(iris_pca, type="l", main="iris 스크리 산점도")   # Scree Plot

# ③ 주성분 분석 절차 #
# 주성분 분석 시각화 : biplot(x, ...)
# → x축을 제1주성분, y축을 제2주성분으로 하고, 각 변수에 대한 주성분 적재값을 화살표로 시각화

iris_pca$loadings
# 제1주성분 점수 : 0.361*Sepal.Length + 0.857*Petal.Length + 0.358*Petal.Width

iris_pca$scores   # 주성분 분석을 통해서 재표현된 모든 iris 관측 데이터의 좌표 확인

biplot(iris_pca, scale=0, main="iris biplot")   # 주성분 점수 시각화화
# 제1주성분에 포함되지 않은 Sepal.WIdth는 Comp.1과 거의 수직





##### <2> 요인 분석 #####


### (1) 요인 분석(Factor Analysis) 개념 ###
# 요인 분석 : 변수 간의 상관관계를 고려하여 서로 유사한 변수들을 묶어 새로운 잠재요인들을 추출하는 분석 방법







##### [2] 정형 데이터 분석 #####





##### <1> 분류 모델 #####

# ② 회귀 분석 가정 #
# 선형성 / 독립성 / 등분산성 / 비상관성 / 정상성

# ④ 회귀 분석 평가(검증) #
# 적합도 검정 : 다변량 회귀 분석에서는 독립젼수의 수가 많아지면 결정계수(R^2)가 높아지므로
#               독립변수가 유의하든, 유의하지 않든 독립변수의 수가 많아지면 결정계수가 높아지는 단점
#               → 조정된 R^2 확인
# 다중공선성(Multicollinearity) 검사 방법 : 분산팽창 요인, 상태지수
#                                           → 분산팽창 요인(VIF : Variation  Inflation Factor)
#                                             : 다중 회귀 모델에서 독립변수 간 상관관계가 있는지 측정하는 척도

# ⑤ 회귀 분석 모형 결과 활용 #
# plot(x, which) : x에는 선형 회귀 모형이 들어가야 함 / which : 그래프의 종류 지정(6가지 종류)
# ----- #
# x <- lm(Salary~PutOuts, data=Hitters)
# plot(x, which=c(1:6))
# ----- #
# 그래프 유형 : 적합값 대비 잔차 그래프(Reisiduals vs Fitted) / 잔차의 Q-Q 그래프(Normal Q-Q) /
#               표준화 잔차 그래프(Scale-Location) / 쿡의 거리(Cook's distance) /
#               레버리지 대비 잔차 그래프(Residuals vs Leverage) / 레버리지 대비 쿡의 거리(Cook's distance vs Leverage)

# 단순 선형 회귀 분석 수행 예제
install.packages("ISLR")
library(ISLR)
summary(lm(Salary~PutOuts, data=Hitters))
# 회귀식 : Salary = 0.48423*PutOuts + 395.15532

# 다중 선형 회귀 분석 수행 예제
str(Hitters)
head(Hitters)
summary(Hitters)   # Salary에 59개의 NA...

hitters <- na.omit(Hitters)   # 결측값 제거
summary(hitters)   # Salary의 NA 제거

full_model <- lm(Salary~., data=hitters)
summary(full_model)
# P-Value가 유의수준 0.05보다 큰 변수들을 제거
# DivisionW라는 더미 변수가 자동으로 생성

first_model <- lm(Salary~AtBat+Hits+Walks+CWalks+Division+PutOuts,
                  data=hitters)
fit_model <- step(first_model, direction="backward")

# vif 함수를 사용하기 위해서는 car 패키지 설치 필요
install.packages("car")
library(car)
vif(fit_model)   # AtBat과 Hits 변수에서 VIF가 10보다 크기 때문에 다중공선성 문제가 심각한 것으로 해석

# VIF 수치가 가장 높은 AtBat을 제거한 후 모형을 생성한 후에 다시 다중공선성 문제를 확인
second_model <- lm(Salary~Hits+CWalks+Division+PutOuts,
                   data=hitters)
vif(second_model)   # 다중공선성 문제 해결

summary(second_model)


### (2) 로지스틱 회귀 분석 ###

# ① 로지스틱 회귀 분석 개념 ###
# 사후 확률(Posterior Probability) : 모형의 적합을 통해 추정된 확률

# ② 변수 선택 #
# 변수의 유의성 검정 : Z-통계량 / 모형의 유의성 검정 : 카이제곱 검정

# ③ 로지스틱 회귀 분석 모형 구축 #
# glm(formula, family, data, ...) → family : 모델에서 사용할 분포(ex. binomial)

# ④ 로지스틱 회귀 분석 모형 평가 #

# 혼동 행렬(Confusion Matrix)
# confusionMatrix(data, reference, ...)
# → data : 예측된 분류 데이터 또는 분할표(table 형식), reference : 실제 분류 데이터

# AUC(AUROC : Area Under ROC)
# ★ROC 곡선의 x축은 FPR(False Positive Ratio), y축은 TPR(True Positive Ratio)로 두고
# 아랫부분의 면적인 AUC를 기준으로 모형을 평가
# auc(actual, predicted, ...)
# → actual : 정답인 label의 벡터(numeric, character 또는 factor), predicted : 예측된 값의 벡터
# auc 함수를 처음 사용할 경우 ModelMetrics 패키지 설치 필요
install.packages("ModelMetrics")
library(ModelMetrics)

install.packages("ISLR")
library(ISLR)
str(Default)
head(Default)
summary(Default)   # 결측값 없음

# 분석 모형 구축 - 유의성 검정
library(ISLR)
bankruptcy <- Default
set.seed(202012)           # 동일 모형 생성을 위한 seed 생성
train_idx <- sample(
  1:nrow(bankruptcy),
  size=0.8*nrow(bankruptcy),
  replace=FALSE
)
test_idx <- (-train_idx)   # train_idx를 제외하고 test_idx 생성
# ----- #
1:nrow(bankruptcy)
train_idx
test_idx
cat("length of train_idx :", length(train_idx))
cat("length of test_idx  :", length(test_idx))
# length(train_idx)   # ★length 함수는 열의 개수...;;
# length(test_idx)
cat("dimension of train_idx :", dim(train_idx))   # NULL
cat("dimension of test_idx  :",dim(test_idx))     # NULL
# ----- #
bankruptcy_train <- bankruptcy[train_idx, ]
bankruptcy_test <- bankruptcy[test_idx, ]
View(bankruptcy_train)
cat("dimension of bankruptcy_train :", dim(bankruptcy_train))
cat("dimension of bankruptcy_test  :", dim(bankruptcy_test))

full_model <- glm(default~.,
                  family=binomial,
                  data=bankruptcy_train)

# 분석 모형 구축 - step 함수 이용
step_model <- step(full_model, direction="both")
# default ~ student + balance

# 분석 모형 구축 - 변수의 유의성 검정
summary(step_model)   # Null deviance : 2354.0, Residual deviance : 1287.4
# studentYes와 balance는 P-Value가 유의수준 0.05보다 작으므로 유의미한 변수...!!

# 분석 모형 구축 - 모형의 유의성 검정
null_deviance <- 2354.0                               # Null deviance : 독립변수가 없는 모형의 이탈도
residual_deviance <- 1287.4                           # Residual deviance : 선택된 모형의 이탈도
model_deviance <- null_deviance - residual_deviance   # Null deviance와 Residual deviance의 값으로 카이제곱 검정 실시
pchisq(model_deviance,
       df=2,   # 자유도는 선택된 변수의 개수
               # Null deviance의 자유도가 7999이고, Residual deviance의 자유도가 7997이므로
       lower.tail=FALSE)
# 2.458968e-232 → 자유도(df)가 2인 카이제곱 분포의 확률변수가 model_deviance일 때 누적분포 함수의 값

# 분석 모형 구축 - 다중공선성 확인
install.packages("car")
library(car)
vif(step_model)   # VIF가 4를 초과하는 값이 없으므로 다중공선성 문제가 없다고 판단

# 분석 모형 평가 - 평가용 데이터를 이용한 분류
pred <- predict(step_model,
                newdata=bankruptcy_test[, -1],   # 첫 번째 열을 제외
                type="response")
df_pred <- as.data.frame(pred)
df_pred$default <- ifelse(df_pred$pred>=0.5,
                          df_pred$default <- "Yes",
                          df_pred$default <- "No")
df_pred$default <- as.factor(df_pred$default)

# 분석 모형 평가 - 혼동 행렬
install.packages("caret")
library(caret)
confusionMatrix(data=df_pred$default,
                reference=bankruptcy_test[ , 1])
# Kappa : 0.4439 → 카파 통계량은 0.4439로서 모형은 보통의 일치도를 보임

# 분석 모형 평가 - AUC
library(ModelMetrics)
auc(actual=bankruptcy_test[ , 1], predicted=df_pred$default)
# 0.6481792 → 0.6과 0.7 사이이므로 불량(Poor)의 성능을 보임


### (3) 의사결정나무(Decision Tree) ###

# ① 의사결정나무 개념 #
# 의사결정나무 기법의 해석이 용이한 이유는 계산 결과가 의사결정나무에 직접 나타나기 떄문

# ② 의사결정나무 분석 함수 종류 #
# rpart() : CART 기법 사용 / tree() : 불순도의 측도로 엔트로피 지수 사용 / ctree()
# CART 기법 : 각 독립변수를 이분화하는 과정을 반복하여 이진 트리 형태를 형성함으로써 분류를 수행하는 방법
# → 불순도의 측도로 출력(목적)변수가 범주형일 경우는 지니 지수를 이용하고,
#   연속형일 경우는 분산을 이용한 이진 분리(Binary Split)를 이용

str(iris)
head(iris)
summary(iris)

# 분석 모형 구축
library(rpart)
md <- rpart(Species~., data=iris)   # iris 데이터를 rpart 함수를 이용해 호출
md

# 시각화 → 시험 환경에서는 사용 불가...;;
plot(md, compress=TRUE, margin=0.5)
text(md, cex=1)

install.packages("rpart.plot")
library(rpart.plot)
windows(height=10, width=12)
prp(md, type=2, extra=2)   # type : 트리 표현, extra : 노드의 추가 정보 표기

# 분석 모형 평가
ls(md)   # 저장된 변수 확인
md$cptable   # 가지치기 및 트리의 최대 크기를 조절하기 위해 cptable을 사용
# CP : 복잡성 / nsplit : 가지의 분기 수 / rel error : 오류율 / xerror : 교차 검증 오류 / xstd : 교차 검증 오류의 표준오차
plotcp(md)

tree_pred <- predict(md, newdata=iris, type="class")
library(caret)
confusionMatrix(tree_pred, reference=iris$Species)


### (4) 서포트 벡터 머신(SVM : Support Vector Machine) ###
# 서포트 백터 머신 : 데이터를 분리하는 초평면(Hyperplane) 중에서 데이터들과 거리가 가장 먼 초평면을 선택하여 분리하는
#                    지도 학습 기반의 이진 선형 분류 모델
#                    → 최대 마진(Margin : 여유 공간)을 가지는 비확률적 선형 판별에 기초한 이진 분류기
# svm(formula, data=NULL)
# predict(object, data, type)   # type → response : 예측값, probabilities : 확률

str(iris)
head(iris)
summary(iris)

# 분석 모형 구축
# svm 함수를 처음 사용할 경우 e1071 패키지 설치 필요
install.packages("e1071")
library(e1071)
model <- svm(Species~., data=iris)
model

# 분석 모형 평가
pred <- predict(model, iris)
library(caret)
confusionMatrix(data=pred, reference=iris$Species)


### (5) K-NN(K-최근접 이웃 : K-Nearest Neighbor) ###
# K-NN 알고리즘 : 새로운 데이터 클래스를 해당 데이터와 가장 가까운 k개 데이터들의 클래스로 분류하는 알고리즘
# knn(train, test, cl, k) → cl : 훈련 데이터의 종속변수, k : 근접 이웃의 수(default=1)

# knn 함수를 처음 사용할 경우 class 패키지 설치 필요
install.packages("class")
library(class)

data <- iris[ , c("Sepal.Length", "Sepal.Width", "Species")]
set.seed(1234)
View(data)

# 변수 선택
idx <- sample(x=c("train", "valid", "test"),
              size=nrow(data),
              replace=TRUE,
              prob=c(3, 1, 1))
# View(idx)
train <- data[idx=="train", ]   # 모든 열을 선택하기 위해 뒤에 ","
valid <- data[idx=="valid", ]
test <- data[idx=="test", ]

train_x <- train[ , -3]         # 3번쨰 열(Species) 제외
valid_x <- valid[ , -3]
test_x <- test[ , -3]

train_y <- train[ , 3]          # 3번째 열(Species) y로 선택
valid_y <- valid[ , 3]
test_y <- test[ , 3]

knn_1 <- knn(train=train_x,
             test=valid_x,
             cl=train_y,
             k=1)
knn_2 <- knn(train=train_x,
             test=valid_x,
             cl=train_y,
             k=2)               # k를 변경하면서 분류 정확도가 가장 높은 k 탐색
                                # for문 사용

# knn 함수 적용 결과 분류 정확도가 가장 높은 k를 선택하는 과정
accuracy_k <- NULL
for (i in c(1:nrow(train_x))) {
  set.seed(1234)
  knn_k <- knn(train=train_x,
               test=valid_x,
               cl=train_y,
               k=i)
  # 분류 정확도 산정
  accuracy_k <- c(accuracy_k,
                  sum(knn_k==valid_y) / length(valid_y))
}
accuracy_k
sum(knn_k==valid_y)
length(valid_y)
sum(knn_k==valid_y) / length(valid_y)
# 최적의 분류 정확도 선정
valid_k <- data.frame(k=c(1:nrow(train_x)),
                      accuracy=accuracy_k)
View(valid_k)
plot(formula=accuracy~k,
     data=valid_k,
     type="o",
     pch=20,
     main="validation - optimal k")

min(valid_k[valid_k$accuracy %in%
              max(accuracy_k), "k"])   # 분류 정확도가 가장 높으면서 가장 작은 k 출력
max(accuracy_k)
# 분류 정확도가 가장 높으면서 가장 작은 k값 : 13 / 최적의 분류 정확도 : 0.8888889

# 모형 평가
knn_13 <- knn(train=train_x,
              test=test_x,
              cl=train_y,
              k=13)
library(caret)
confusionMatrix(knn_13, reference=test_y)


### (6) ANN(Artificial Neural Network : 인공 신경망) ###
# nnet(formula, data, size, maxit, decay=5e-04, ...)

data(iris)
iris.scaled <- cbind(scale(iris[-5]),
                     iris[5])   # ★iris의 Species를 제외한 변수에 scale() 함수 적용 후
                                # cbind로 다시 Species를 포함한 iris.scaled 변수 생성
set.seed(1000)
index <- c(sample(1:50, size=35),
           sample(51:100, size=35),
           sample(101:150, size=35))
train <- iris.scaled[index, ]
test <- iris.scaled[-index, ]

# 분석 모형 구축
set.seed(1234)
# nnet 함수를 처음 사용할 경우 nnet 패키지 설치 필요
install.packages("nnet")
library(nnet)
model.nnet <- nnet(Species~.,
                   data=train,
                   size=2,   # 은닉층 : 2
                   maxit=200,   # 최대 반복 횟수 : 200
                   decay=5e-04)   # 가중치 감소의 모수 : 5e-04
# weights : 19 / 초깃갑 : 132.822952 / 최종값 : 6.088585

summary(model.nnet)


### (7) 나이브 베이즈(Naive Bayes) 기법 ###
# 나이브 베이즈 분류 : 특성들 사이의 독립을 가정하는 베이즈 정리를 적용한 확률 분류기
# ★베이즈 정리(Bayes' Theorem) : 어떤 사건에 대해 관측 전(사전 확률) 원인에 대한 가능성과
#                                관측 후(사후 확률)의 원인 가능성 사이의 관계를 설명하는 확률 이론
# naiveBayes(formula, data, subset, laplace, ...)
# → subset : 훈련 데이터에서 사용할 데이터를 지정하는 인덱스 벡터
#   laplace : 라플라스 추정기 사용 여부(default : 0)
# 라플라스 매개변수는 라플라스 추정기(Laplace Estimator)로 중간에 0이 들어가서
# 모든 확률을 0으로 만들어 버리는 것을 방지하기 위한 인자

str(iris)
head(iris)
summary(iris)

# 분석 모형 구축
library(e1071)
train_data <- sample(1:150, size=100)

naive_model <- naiveBayes(Species~.,
                          data=iris,
                          subset=train_data)
naive_model
# A-priori probabilities : 훈련 데이터로 알 수 있는 사전 확률 출력
# Conditional probabilities : 데이터의 분포 매개변수로 평균과 표준편차를 출력
#                             → [,1]은 평균, [,2]는 표준편차

# 분석 모형 평가
pred <- predict(naive_model, newdata=iris)
confusionMatrix(pred, reference=iris$Species)


### (8) 앙상블(Ensemble) ###
# 앙상블 : 여러 가지 동일한 종류 또는 서로 상이한 모형들의 예측/분류 결과를 종합하여 최종적인 의사결정에 활용하는 기법

# ② 앙상블 기법의 종류 - 배깅(Bagging : BootStrap Aggregating) #
# 배깅 : 훈련 데이터에서 다수의 부트스트랩(BootStrap) 자료를 생성하고,
#        각 자료를 모델링한 후 결합하여 최종 예측 모형을 만드는 알고리즘
# 부트스트랩 : 주어진 자료에서 동일한 크기의 표본을 랜덤 복원 추출로 뽑은 자료
# bagging(formula, data, mfinal, control) → mfinal : 반복수 또는 트리의 수(default=100), control : 의사결정나무 옵션
# 분석 함수 결과 : trees / votes / prob / class / samples / importance

library(mlbench)
data(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)
head(PimaIndiansDiabetes2)
summary(PimaIndiansDiabetes2)   # 결측값 존재

# 데이터 전처리
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
summary(PimaIndiansDiabetes2)   # 결측값 제거 확인

# 분석 모형 구축
train.idx <- sample(1:nrow(PimaIndiansDiabetes2),
                    size=nrow(PimaIndiansDiabetes2)*2/3)
train <- PimaIndiansDiabetes2[train.idx, ]   # "," 필수
test <- PimaIndiansDiabetes2[-train.idx, ]   # "," 필수

# bagging 함수를 처음 사용할 경우 ipred 패키지 설치 필요
install.packages("ipred")
library(ipred)
md.bagging <- bagging(diabetes~.,
                      data=train,
                      nbagg=25)   # Bagging classification trees with 25 bootstrap replications-!

# 분석 모형 평가
pred <- predict(md.bagging, test)
library(caret)
confusionMatrix(as.factor(pred),
                reference=test$diabetes,
                positive="pos")   # ★

# ③ 앙상블 기법의 종류 - 부스팅(Boosting) #
# 부스팅 : 예측력이 약한 모형(Weak Learner)들을 결합하여 강한 예측 모형을 만드는 방법
# AdaBoost, GBM, LightGBM, XGBoost, CatBoost, ...

# ★ XGBoost(eXtreme Gradient Boosting) 알고리즘 ★ #
# xgb.train(params, data, nrounds, early_stopping_rounds, watchlist)
# xgb.train 함수는 독립변수가 수치형 데이터만 사용이 가능하며,
#                  명목형인 경우에는 One-Hot Encoding을 수행하여 수치형으로 변환한 후 사용
# xgb.DMatrix(data, info, ...)

# 분석 모형 구축
# xgb.train 함수를 사용하기 위해서는 xgboost 패키지 설치 필요
install.packages("xgboost")
library(xgboost)
help(xgboost)   # ★

# 종속변수가 팩터(Factor)형인 경우에는 숫자로 변환한 후 0부터 시작하기 위해 1을 뺌
train.label <- as.integer(train$diabetes)-1
# 훈련 데이터를 xgb.DMatrix로 변환하기 위하여 행렬(Matrix)로 변환
mat_train.data <- as.matrix(train[ , -9])   # train 데이터는 XGBoost에서 사용한 것을 가져옴
# 평가 데이터를 xgb.DMatrix로 변환하기 위하여 행렬(Matrix)로 변환
mat_test.data <- as.matrix(test[ , -9])     # test 데이터는 XGBoost에서 사용한 것을 가져옴

# 훈련 데이터를 xgb.DMatrix로 변환
xgb.train <- xgb.DMatrix(data=mat_train.data,
                         label=train.label)
# 평가 데이터를 xgb.DMatrix로 변환
xgb.test <- xgb.DMatrix(data=mat_test.data)

# ★ 주요 매개변수 설정 ★ #
param_list <- list(booster="gbtree",              # 부스터 방법 설정(gbtree 또는 gblinear)
                   eta=0.001,                     # 학습률(learning rate)
                   max_depth=10,                  # 한 트리의 최대 깊이
                   gamma=5,                       # Information Gain에 페널티를 부여하는 숫자
                   subsample=0.8,                 # 훈련 데이터의 샘플 비율
                   colsample_bytree=0.8,          # 개별 트리를 구성할 때 컬럼의 subsample 비율
                   objective="binary:logistic",   # 목적 함수 지정
                   eval_metric="auc")             # 모델의 평가 함수(regression : rmse, classification : error 등)

md.xgb <- xgb.train(params=param_list,
                    data=xgb.train,
                    nrounds=200,                      # 반복 횟수 200회
                    early_stopping_rounds=10,         # AUC가 10회 이상 증가하지 않을 경우 학습 조기 중단
                    watchlist=list(val1=xgb.train),   # 모형의 성능을 평가하기 위하여 사용하는 xgb.DMatrix 이름
                    verbose=1)
# 반복 횟수를 200회로 지정했으나 22회에서 학습 조기 중단 → 12회의 AUC 이상 증가 X

# 분석 모형 평가
xgb.pred <- predict(md.xgb,
                    newdata=xgb.test)
xgb.pred2 <- ifelse(xgb.pred>=0.5,
                    xgb.pred <- "pos",
                    xgb.pred <- "neg")
xgb.pred2 <- as.factor(xgb.pred2)
library(caret)
confusionMatrix(xgb.pred2,
                reference=test$diabetes,
                positive="pos")   # ★

# ④ 앙상블 기법의 종류 - 랜덤 포레스트(Random Forest) #
# 랜덤 포레스트 : 의사결정나무의 특징인 분산이 크다는 점을 고려하여 배깅과 부스팅보다 더 많은 무작위성을 주어
#                 약학 학습기들을 생성한 후 이를 선형 결합하여 최종 학습기를 만드는 방법
# randomForest(formula, data, ntree, mtry) → ntree : 사용할 트리의 수, mtry : 각 분할에서 랜덤으로 뽑힌 변수의 수
# 분석 함수 결과 : predicted / err.rate / importance
# predicted : Out-of-bag samples에 기초한 예측값 확인
# importance : 지니 지수의 이익(gain) 또는 불확실성의 감소량을 고려한 측도
# Out-of-bag : 한 번도 포함되지 않은 데이터 / 지니 지수 : 노드의 불순도를 나타내는 값

# 분석 모형 구축
install.packages("randomForest")
library(randomForest)
md.rf <- randomForest(diabetes~.,
                      data=train,
                      ntree=100,
                      proximity=TRUE)
md.rf
print(md.rf)

# importance 함수를 이용하여 변수의 중요도 확인 가능
importance(md.rf)   # glucose 변수의 결과가 가장 큰 값으로 중요도가 가장 높다고 할 수 있음

pred <- predict(md.rf, newdata=test)
confusionMatrix(as.factor(pred),
                test$diabetes,
                positive="pos")





##### <2> 군집 모델 #####


### (1) 군집 분석 개념 ###
# 군집 분석 : 관측된 여러 개의 변숫값으로부터 유사성(Similarity)에만 기초하여 n개의 군집으로 집단화하고,
#             형성된 집단의 특성으로부터 관계를 분석하는 다변량 분석 기법


### (3) 군집 분석 결과 ###
# 덴드로그램(Dendrogram) : 군집의 개체들이 결합되는 순서를 나타내는 트리 형태의 구조


### (4) 군집 분석 종류 - 계층적 군집 분석 ###

# ① 분석 모형 구축(군집 간의 거리 측정) #
# dist : 군집 분석에서 거리를 측정해주는 함수 / hclust : 계층적 군집 분석을 수행
# dist(data, method) → method : 거리 측정 방법(euclidean, maximum, manhattan, canberra, binary, minkowsky)
# hclust(data, method) → mothod : 군집 연결 방법(single, complete, average, median, ward.D)

str(USArrests)
head(USArrests)
summary(USArrests)

# 유클리디안 거리 측정
US.dist_euclidean <- dist(USArrests, "euclidean")
US.dist_euclidean
# 맨하탄 거리 측정
US.dist_manhattan <- dist(USArrests, "manhattan")
US.dist_manhattan
# 마할라노비스 거리 측정
mahalanobis(USArrests, colMeans(USArrests), cov(USArrests))   # center : 중심값, cov : 공분산

# 분석 모형 구축(계층적 군집 분석)
# single : 최단거리법, complete : 최장거리법, average : 평균거리법, median : 중심연결법, ward.D : 와드연결법
US.single <- hclust(US.dist_euclidean^2,   # 왜 제곱을 하는거지...?
                    method="single")       # 최단거리법
plot(US.single)
US.complete <- hclust(US.dist_euclidean^2, method="complete")
plot(US.complete)

# ② 군집 분석을 통한 그룹 확인 #
# 덴드로그램 결과에서 가로선을 그었을 때의 세로축 개수를 군집의 수로 선정
# cutree 함수나 rect.hclust 함수를 이용하여 그룹화

group <- cutree(US.single, k=6)   # 6개의 그룹으로 분할
group

rect.hclust(US.single, k=6, border="blue")


### (5) 비계층적 군집 분석 - k-평균(k-means) 군집 분석 ###

# ① k-평균 군집 개념 #
# k개만큼 원하는 군집 수를 초깃값으로 지정하고,
# 각 개체를 가까운 초깃값에 할당하여 군집을 형성하고 각 군집의 평균을 재계산하여 초깃값을 갱신
# 갱신 과정을 반복하여 k개의 최종 군집을 형성

# ② 분석 모형 구축 #
# kmeans(data, centers) → centers : 군집의 개수 설정

# 분석 모형 구축
# wine 데이터 세트를 사용하기 위해서는 rattle 패키지 설치 필요
install.packages("rattle")
library(rattle)
df = scale(wine[-1])
set.seed(1234)
fit.km <- kmeans(df, 3, nstart=25)   # nstart : k-means 알고리즘을 몇 번 실행할지 결정
fit.km$size                          # k개의 점과 가장 가까운 데이터의 개수
fit.km$centers                       # 속성별 k개의 점에 대한 위치

# 분석 모형 활용
plot(df, col=fit.km$cluster)
points(fit.km$center,   # fit.km$centers로 해도 같은 결과
       col=1:3,
       pch=8,
       cex=1.5)





##### <3> 연관 모델 #####


### (1) 연관성 분석(Association Analysis)의 개념 ###
# 연관성 분석 : 데이터 내부에 존재하는 항목 간의 상호관계 혹은 종속관계를 찾아내는 기법
# 장바구니 분석(Market Basket Analysis) or 서열 분석(Sequence Analysis)


### (2) 연관성 분석 함수 ###

# ① as 함수 #
# as(data, class) → class : 연관 분석에서는 "transactions"

# ② inspect 함수 #
# 트랜젝션 데이터가 아닌 데이터는 apriori 함수를 이용하기 전에 트랜잭션 형태로 변경하여 사용해야 함
# inspect(x, ...)

# ③ apriori 함수 #
# apriori(data, parameter, appearance, control)
# → parameter : 최소 지지도(supp), 신뢰도(conf), 최대 아이템 개수(maxien), 최소 아이템 개수(minien)

mx.ex <- matrix(
  c(1, 1, 1, 1, 10,
    1, 1, 0, 1, 0,
    1, 0, 0, 1, 0,
    1, 1, 1, 0, 0,
    1, 1, 1, 0, 0),
  ncol=5,
  byrow=TRUE   # 예제용 5*5 구매 데이터(행렬)
)
mx.ex
rownames(mx.ex) <- c("p1", "p2", "p3", "p4", "p5")
colnames(mx.ex) <- c("a", "b", "c", "d", "e")
mx.ex

# Transaction 클래스로 변환 및 데이터 확인
# as, apriori, inspect 함수 사용을 위해 arules 패키지 설치 필요
install.packages("arules")
library(arules)
trx.ex <- as(mx.ex, "transactions")
trx.ex
summary(trx.ex)
inspect(trx.ex)
#     items           transactionID
# [1] {a, b, c, d, e} p1
# [2] {a, b, d}       p2
# [3] {a, d}          p3
# [4] {a, b, c}       p4
# [5] {a, b, c}       p5

# ----- #

# 데이터 세트 준비
install.packages("arulesViz")
library(arulesViz)
data("Groceries")
summary(Groceries)   # 이미 transaction으로 변환되어 있음

# apriori 함수를 통한 연관 규칙 생성
apr <- apriori(Groceries,
               parameter=list(support=0.01,      # 최소 지지도 : 0.01
                              confidence=0.3))   # 최소 신뢰도 : 0.3
# [125 rule(s)]를 통해 총 125개의 연관 규칙이 생성된 것을 확인
# Parameter specification을 통해 구체적인 매개변수 값을 확인 가능
# 규칙의 수에 따라 지지도와 신뢰도를 높이거나 낮추어 규칙 조정 가능

# inspect 함수를 통해 연관 규칙 확인
inspect(sort(apr, by="lift")[1:10])   # lift(향상도)
# 1번 규칙을 통해 citrus fruit, other vegetables와 root vegetables에 대한 향상도가 3.295045로 가장 높음을 확인
# 이를 통해 묶음 상품으로 같이 구성하거나, 진열 위치를 조정하는 것과 같이 분석 결과 활용 가능
