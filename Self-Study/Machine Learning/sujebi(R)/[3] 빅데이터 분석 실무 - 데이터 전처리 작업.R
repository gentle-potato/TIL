# *** CH 02 데이터 전처리 작업 *** #







##### [1] 데이터 전처리 패키지 #####





##### <2> 데이터 전처리 패키지 유형 #####


### (1) plyr 패키지 ###
# plyr 패키지를 처음 사용할 경우 설치 필요
install.packages("plyr")
library(plyr)
# plyr 패키지 함수는 **ply 혀태이고,
# 첫 번째 글자는 입력 데이터의 형태, 두 번쩨 글자는 출력 데이터의 형태를 의미
# data frame / list / array

# ① adply 함수 #
# adply(data, margins, fun)   # margins : 1이면 행 방향, 2면 열 방향
adply(
  iris,
  1,
  function(row){
    row$Sepal.Length>=5.0 &
      row$Species=="setosa"
  }
)

# ② ddply 함수 #
# ddply(data, .variables, ddply-func, fun)
ddply(iris, .(Species, Petal.Length<1.5), function(sub){
  data.frame(
    mean_to=mean(sub$Sepal.Length), mean_so=mean(sub$Sepal.Width),
    mean_wo=mean(sub$Petal.Length), mean_jo=mean(sub$Petal.Width)
  )
})

ddply(iris, .(Species), summarise, mean_to=mean(Sepal.Length))
# summarise 쓰는 것과 안 쓰는 것의 차이를 잘 모르겠다...;

# ③ transform 함수 #
# transform(_data, tag1=value1, tag2=value2, ...)
transform(iris, Total.w=Sepal.Width+Petal.Width)


### (2) dplyr 패키지 ###
# dplr 패키지를 처음 사용할 경우 설치 필요
install.packages("dplyr")
library(dplyr)
# %>% : '파이프 연산자(Pipe Operator)라고 읽고, 함수들을 연결하는 기능을 함
# 그 앞에 나온 데이터를 계속해서 사용하겠다는 의미
# dplyr 패키지는 %>% 기호를 이용해서 함수를 나열하는 방식으로 코드를 작성

# ① select 함수 #
iris %>% select(Sepal.Length)

# ② filter 함수 #
iris %>% filter(Species=='setosa') %>% select(Sepal.Length, Sepal.Width)

# ③ mutate 함수
# 파생변수 생성
iris %>%
  filter(Species=='virginica') %>%
  mutate(Len=ifelse(Sepal.Length>6, 'L', 'S')) %>% select(Species, Len)

# ④ group_by와 summarise 함수 #
iris %>%
  group_by(Species) %>%
  summarise(Petal.Width=mean(Petal.Width))

# ⑤ arrange 함수 #
iris %>%
  filter(Species=='setosa') %>%
  mutate(Len=ifelse(Sepal.Length>5, 'L', 'S')) %>%
  select(Species, Len, Sepal.Length) %>%
  arrange(desc(Sepal.Length))

# ⑥ inner_join 함수 #
X <- data.frame(Department=c(11, 12, 13, 14),
                DepartmentName=c("Production", "Sales", "Marketing", "Research"),
                Manager=c(1, 4, 5, NA))
X

Y <- data.frame(Emplyee=c(1, 2, 3, 4, 5, 6),
                EmployeeName=c("A", "B", "C", "D", "E", "F"),
                Department=c(11, 11, 12, 12, 13, 21),
                Salary=c(80, 60, 90, 100, 80, 70))
Y

inner_join(X, Y, by="Department")

# ⑦ left_join 함수 #
left_join(X, Y, by="Department")

# ⑧ right_join 함수 #
right_join(X, Y, by="Department")

# ⑨ full_join 함수 #
full_join(X, Y, by="Department")

# ⑩ bind_rows 함수 #
x <- data.frame(x=1:3, y=1:3)
x

y <- data.frame(x=4:6, z=4:6)
y

bind_rows(x, y)

# ⑪ bind_cols 함수 #
x <- data.frame(title=c(1:5),
                a=c(30, 70, 45, 90, 65))
x

y <- data.frame(b=c(70, 65, 80, 80, 90))
y

bind_cols(x, y)


### (3) reshape2 패키지 ###
# reshape2 패키지를 사용하기 위해서는 패키지 설치 필요
install.packages("reshape2")
library(reshape2)


# ① melt 함수 #
# Cars93 데이터 세트를 사용하기 위해서는 MASS 패키지 설치 필요
install.packages("MASS")
library(MASS)

a <- melt(data=Cars93,
          id.vars=c("Type", "Origin"),
          measure.vars=c("MPG.city", "MPG.highway"))
a

View(Cars93)

# ② dcast 함수 #
a <- melt(data=Cars93,
          id.vars=c("Type", "Origin"),
          measure.vars=c("MPG.city", "MPG.highway"))
a
dcast(data=a, Type+Origin~variable, fun=mean)   # dcast 함수 적용


### (4) data.table 패키지 ###
# data.table 패키지를 처음 사용할 경우 설치 필요
install.packages("data.table")
library(data.table)

# ① 데이터 테이블 생성 #
t <- data.table(x=c(1:3),
                y=c("가", "나", "다"))
t

# ② 데이터 테이블 변환 #
iris_table <- as.data.table(iris)
iris_table

# ③ 데이터 접근 #
iris_table <- as.data.table(iris)

iris_table[1, ]

iris_table[c(1:2), ]

iris_table[ , mean(Petal.Width), by=Species]







####### [2] 데이터 정제 #######





##### <1> 결측값 #####


### (2) 결측값 인식 ###

ds_NA <- head(airquality, 5)
ds_NA

is.na(ds_NA)
complete.cases(ds_NA)   # 행별로 결측값 존재 여부(없으면 TRUE / 있으면 FALSE)

# ----- #
# summary 함수를 이용한 결측값 확인
install.packages("mlbench")
library(mlbench)
data(PimaIndiansDiabetes2)
Pima2 <- PimaIndiansDiabetes2
Pima2

str(Pima2)   # 데이터 세트 속성 확인

summary(Pima2)

complete.cases(Pima2)
is.na(Pima2)
sum(is.na(Pima2))
sum(!complete.cases(Pima2))   # 768개의 행 중에서 376개의 행에서 결측값 존재
colSums(is.na(Pima2))         # 컬럼별 결측값의 수
# ----- #


### (3) 결측값 처리 ###

# ① 결측값 삭제(특정 컬럼 삭제) #
Pima2$insulin <- NULL
Pima2$triceps <- NULL
sum(!complete.cases(Pima2))
colSums(is.na(Pima2))

# ② 단순대치법 #

# 완전분석법(Complete Analysis)
library(dplyr)
Pima3 <- Pima2 %>% filter(!is.na(glucose) & !is.na(mass))
colSums(is.na(Pima3))

dim(Pima3)   # 전체 행의 수 확인

Pima4 <- na.omit(Pima3)
colSums(is.na(Pima4))

dim(Pima4)

# 평균대치법(Mean Imputation)
head(ds_NA$Ozone)
ds_NA$Ozone <- ifelse(   # ifelse로 결측값 평균대치
  is.na(ds_NA$Ozone),
  mean(ds_NA$Ozone, na.rm=TRUE),
  ds_NA$Ozone
)
table(is.na(ds_NA$Ozone))

ds_NA2 <- head(airquality, 5)
ds_NA2
ds_NA2[is.na(ds_NA2$Ozone), "Ozone"] <-
  mean(ds_NA2$Ozone, na.rm=TRUE)
table(is.na(ds_NA2$Ozone))

# ----- #
summary(Pima3)

mean_press <- mean(Pima3$pressure, na.rm=TRUE)
mean_press

std_before <- sd(Pima3$pressure, na.rm=TRUE)
std_before

Pima3$pressure <- ifelse(is.na(Pima3$pressure), mean_press, Pima3$pressure)
std_after <- sd(Pima3$pressure)
std_after

std_diff <- std_after - std_before
print(std_diff)
# ----- #





##### <2> 이상값 #####


### (2) 이상값 판별 ###

# ① ESD(Extreme Studentized Deviation) #
score <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100000000)
name <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
df_score <- data.frame(name, score)

esd <- function(x){
  return(abs(x-mean(x))/sd(x)<3)
}
esd(score)                        # L만 FALSE

library(dplyr)
df_score %>% filter(esd(score))   # L만 제외 추출

# ② 사분위수 범위(Q3-Q1) #

# 박스 플롯(box plot)
score <- c(65, 60, 70, 75, 200)
name <- c("Bell", "Cherry", "Don", "Jake", "Fox")
df_score <- data.frame(name, score)

box_score <- boxplot(df_score$score)
box_score$out
box_score$stats
min_score <- box_score$stats[1]
max_score <- box_score$stats[5]

library(dplyr)
df_score %>% filter(score>=min_score & score<=max_score)

# IQR 함수
score <- c(65, 60, 70, 75, 200)
name <- c("Bell", "Cherry", "Don", "Jake", "Fox")
df_score <- data.frame(name, score)
min_score <- median(df_score$score)-2*IQR(df_score$score)
max_score <- median(df_score$score)+2*IQR(df_score$score)

library(dplyr)
df_score %>% filter(score>=min_score & score<=max_score)


### (3) 이상값 처리 ###
# 삭제, 대체, 변환, 분류 등







##### [3] 데이터 변환 #####





##### <1> 데이터 유형 변환 #####


### (1) as.character 함수 ###

a <- 0:9
a
str(a)
a <- as.character(a)
a
str(a)


### (2) as.numeric 함수 ###

a <- 0:9
a <- as.character(a)
a <- as.numeric(a)
a
str(a)


### (3) as.double 함수 ###

a <- 0:9
a <- as.double(a)
a
typeof(a)


### (4) as.logical 함수 ###

a <- 0:9
a <- as.logical(a)
a
str(a)


### (5) as.integer 함수 ###

a <- 10
typeof(a)
a <- as.integer(a)
typeof(a)

a <- 0:9
a <- as.logical(a)
a <- as.integer(a)
a
str(a)





##### <2> 자료구조 변환 ######


### (1) as.data,frame 함수 ###

a <- 0:4
str(a)
a <- as.data.frame(a)
a
str(a)


### (2) as.list 함수 ###

a <- 0:9
a <- as.data.frame(a)
a <- as.list(a)
a


### (3) as.matrix 함수 ###

a <- 0:4
a <- as.matrix(a)
a
str(a)


### (4) as.vector 함수 ###

a <- 0:9
a <- as.vector(a)
a
str(a)


### (5) as.factor 함수 ###

a <- 0:9
a <- as.factor(a)
a
str(a)





##### <3> 데이터의 범위 변환 #####


### (1) 정규화(Normalization) ###

# ② 최소-최대 정규화(Min-Max Normalizaion) #
# 단점 : 이상값에 너무 많은 영향을 받음

data <- c(1, 3, 5, 7, 9)
data_minmax <- scale(data,
                     center=1,
                     scale=8)
data_minmax
mode(data)
class(data)

a <- 1:10
a
normalize <- function(a){
  return ((a-min(a))/(max(a)-min(a)))
}
normalize(a)
scale(a)
as.vector(scale(a))   # zero-centered data로 만듦


### (2) 표준화(Standardization) ###

# ② Z-스코어(Z-Score) #
# 이상치(Outlier) 문제를 피하는 기법

data <- c(1, 3, 5, 7, 9)
data_zscore <- scale(data)
mean(data_zscore)
sd(data_zscore)
data_zscore







##### [4] 표본 추출 및 집약처리 #####





##### <1> 표본 추출 #####


### (1) 표본 추출법 ###
# 단순 무작위 추출 / 계통 추출 / 층화 추출 / 군집 추출


### (2) 표본 추출 함수 ###

# ① sample 함수 #
# sample(x, size, replace, prob)