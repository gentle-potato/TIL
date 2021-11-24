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
