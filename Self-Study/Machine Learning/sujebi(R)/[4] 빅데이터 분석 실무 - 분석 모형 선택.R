# *** CH 03 분석 모형 선택 *** #







##### [1] 데이터 탐색 #####





##### <2> 데이터 탐색 방법 #####


### (1) 개별 데이터 탐색 ###

# ① 범주형 데이터 #

# 빈도수 탐색
table(mtcars$cyl)

# 백분율 및 비율 탐색
cnt <- table(mtcars$cyl)
total <- length(mtcars$cyl)
cnt/total
# 별해
prop.table(cnt)

# 시각화
cnt <- table(mtcars$cyl)
barplot(cnt,
        xlab="기통",
        ylab="수량",
        main="기통별 수량")

cnt <- table(mtcars$cyl)
pie(cnt,
    main="기통별 비율")

# ② 수치형 데이터 #

# 요약 통계량
summary(mtcars$wt)   # 요약 통계량 출력
head(mtcars$wt)      # 데이터 앞부분 출력
tail(mtcars$wt)      # 데이터 뒷부분 출력
str(mtcars)          # 데이터의 속성을 출력
View(mtcars)         # 뷰어 창에서 데이터 확인
dim(mtcars)          # 데이터의 차원(행, 열)을 출력

# 개별 데이터의 시각화
wt_hist <- hist(mtcars$wt,
                breaks=5,             # 5개의 구간
                xlab="무게",
                ylab="수량",
                main="무게별 수량")
wt_box <- boxplot(mtcars$wt,
                  main="무게 분포")


### (2) 다차원 데이터 탐색 ###

# ① 범주형-범주형 데이터 탐색 #

# 빈도수와 비율 탐색 - table
table(mtcars$am, mtcars$cyl)
mtcars$label_am <- factor(mtcars$am,
                          labels = c("automatic", "manual"))
table(mtcars$label_am, mtcars$cyl)

# 빈도수와 비율 탐색 - prop.table, addmargins
prop_table <- prop.table(table(mtcars$label_am, mtcars$cyl)) * 100
addmargins(round(prop_table, digits=1))   # round 함수에서 digits : 소수점 자릿수

# 빈도수와 비율 시각화
barplot(table(mtcars$label_am, mtcars$cyl),
        xlab="실린더 수",
        ylab="수량")

# ② 수치형-수치형 데이터 탐색 #

# 상관관계 탐색
# cor(x, y, method)   # method : pearson, spearman, kendall
cor_mpg_wt <- cor(mtcars$mpg,
                  mtcars$wt)
cor_mpg_wt

# 상관관계 시각화
plot(mtcars$mpg, mtcars$wt)

# ③ 범주형-수치형 데이터 탐색 #

# 그룹 간의 기술 통계량 분석
# aggregate(formula, data, FUN, ...)
aggregate(mpg~cyl,
          data=mtcars,
          FUN=mean)

# 그룹 간의 시각화
boxplot(mpg~cyl, data=mtcars, main="기통별 연비")







##### [2] 상관 분석 #####





##### <2> 상관관계의 표현 방법 #####


### (1) 산점도(Scatter Plot)를 통한 표현 방법 ###


### (2) 공분산(Covariance)을 통한 표현 방법 ###

# 상관계수 시각화 함수
# pairs(x, ...), corrplot 함수

# 상관계수 가설검정
# cor.test(x, y method)   # method : pearson, spearman, kendall

install.packages("mlbench")
library(mlbench)

data(PimaIndiansDiabetes)
df_pima <- PimaIndiansDiabetes[c(3:5, 8)]   # 3~5, 8번째 열

# ----- #
df_pima
PimaIndiansDiabetes
# ----- #

str(df_pima)
summary(df_pima)   # 결측치 없음 확인

cor(x=df_pima, method="pearson")    # insulin과 triceps의 뚜렷한 양의 상관관계
cor(x=df_pima, method="spearman")   # insulin과 triceps의 뚜렷한 양의 상관관계
cor(x=df_pima, method="kendall")    # insulin과 triceps의 뚜렷한 양의 상관관계

windows(width=12, height=10)   # 새로운 윈도우에서 시각화 표시
# corrplot 함수를 처음 사용할 경우 설치 필요
install.packages('corrplot')
library(corrplot)
corrplot(cor(df_pima), method="circle", type="lower")

# 귀무가설(H0) : 변수 1과 변수 2 사이에는 상관관계가 없다. (상관계수 = 0)
# 대립가설(H1) : 변수 1과 변수 2 사이에는 상관관계가 있다. (상관계수 != 0)

# 상관계수 검정 - 정규성 만족 여부 검정 : 샤피로-윌크 검정 함수
# 귀무가설 : 정규분포를 따른다.
shapiro.test(df_pima$triceps)   # 귀무가설 기각
shapiro.test(df_pima$insulin)   # 귀무가설 기각
# 따라서 두 변수 모두 정규성을 만족한다고 할 수 없다. → 비모수 검정인 Spearman or Kendall 이용

cor.test(df_pima$triceps, df_pima$insulin, method="kendall")   # 귀무가설 기각
# insulin과 triceps의 상관계수는 0이 아니라고 유의하게 이야기할 수 있다.
# 상관계수인 tau는 약 0.42로서 뚜렷한 양의 선형관계를 나타낸다.







##### <3> 변수 선택 #####


### (3) 변수 선택 방법 ###

# ① step 함수 #