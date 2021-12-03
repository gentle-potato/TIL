##### [1] 기본 문법 #####





##### <1> 주석 #####


# 주석 내용
comment = "주석
           주석"



##### <2> 도움말 #####

# help(topic)
help(sum)



##### <3> 연산자 #####


### (1) 산술 연산자 ###
7+4
7/4
7%%4
7%/%4
7^4
7**4


### (2) 관계 연산자 ###
7>4
7<4
7==4
7!=4


### (3) 논리 연산자 ###
TRUE & FALSE
3 & 0
TRUE | FALSE
3 | 0
!TRUE
!1
!0
!-1

### (4) 대입 연산자 ###
# =, <-, <<-, ->, ->>

### (5) 기타 연산자 ###
?list
??list
1:7



##### <4> 변수(Variable) #####


### (1) 변수 생성 ###
a = "bigdata"
b <- 5


### (2) 변수명 생성 규칙 ###
# - 변수명은 문자, 숫자, 언더바(_), 마침표(.)를 조합해서 생성
# - 변수명 첫 글자는 마침표 또는 문자로 시작해야 하며, 마침표로 시작할 경우에는 뒤에 숫자가 올 수 X
# - 대시(-)는 변수명으로 사용 불가



##### <5> 데이터 타입 #####


### (2) 데이터 타입 확인 함수 ###

# ① mode 함수 #
# 객체의 형식인 numeric, character, logical 중 하나의 값을 출력하는 함수

# ② typeof #
# mode 함수를 사용했을 경우 numeric으로 출력되는 값 중에 정수형일 경우 integer, 실수형일 경우 double로 반환되고,
# 나머지는 mode 함수와 동일하게 출력하는 함수


### (3) 데이터 기본 타입 ###

# 숫자형numeric #
mode(5)
typeof(5)
typeof(5L)   # R에서 정숫값을 나타내기 위해서는 정수 뒤에 L을 붙여야 함
mode(2.5)
typeof(2.5)

# 문자형character #
mode('a')
typeof('a')
mode('abc')
typeof('abc')

# 논리형logical #
mode(TRUE)
typeof(TRUE)


### (4) 데이터의 값 ###
a <- NA
is.na(a)
b <- 3
is.na(b)


### (5) 데이터 타입 확인 ###
is.numeric(c(1:4))
is.integer(c(1:4))
is.double(c(1:4))
is.character(c("a", "b"))
is.logical(c(1:4))
is.null(c())
is.na(c(NA, NA, 2, NA))



##### <6> 객체 #####


### (2) 벡터Vector ###

# ② 백터 문법 #
# 벡터 생성 함수
x <- c(1, 2, 3, 4)
x
y <- c("1", 2, "3")   # 다른 타입의 데이터를 섞어서 벡터에 저장하면 이들 데이터는 한 가지 타입으로 형태가 변환
y
z <- 5:10
z
# 벡터 반복 함수
# rep(x, times, ...)
x <- rep(c(1, 2, 3), 3)
# 벡터 수열 함수
# seq(from, to, by)
seq(1, 10, 3)
# 벡터의 인덱싱
# ★R 언어는 첫 번째 값이 1번지이니 헷갈리지 않도록 주의-!
x <- c(3:8)
x
x[4]
x[-4]    # 4번째 요소 제외한 모든 요소 출력
x[x>5]   # 조건문
x[5:8]
# 벡터 연산
# 저장하고 있는 원소의 개수가 서로 다른 벡터끼리의 연산을 수행할 때는 원소의 개수가 적은 벡터를
# 부족한 자릿수가 채워질 때까지 반복한 후 계산
x <- c(1:3)
x*3
y <- c(3:5)
x+y
x*y
# 벡터 함수
# trunc(x) : 소수점 버림 / sd(x) : 표준편차 / cor() : 상관계수
# union(x, y) : 합집합 / intersect(x, y) : 교집합 / setdiff(x, y) : 차집합(방향 고려 필요)


### (3) 리스트List ###
# (키, 값) 형태로 데이터를 저장하는 R의 모든 객체를 담을 수 있는 데이터 구조

# ②  리스트 문법 #
# list(key1=value1, key2=value2, ...)
list(name="soo", height=90)
list(name="soo", height=c(2, 6))
list(x=list(a=c(1, 2, 3)),
     y=list(b=(c(1, 2, 3, 4))))


### (4) 행렬Matrix ###
# 2차원의 벡터로 행(row)과 열(column)로 구성된 데이터 타입

# ② 행렬 생성 문법 #
# matrix(data, nrow, ncol, byrow, dimnames)
matrix(c(1:12), nrow=4)
matrix(c(1:12), ncol=4)
matrix(c(1:12), nrow=3, byrow=TRUE)
matrix(c(1:9), nrow=3, dimnames=list(c("t1", "t2", "t3"),
                                     c("c1", "c2", "c3")))

# ③ 행렬 함수 #
# dim(x) / dim(x) <- c(m, n) / nrow(x) / ncol(x)
# x[m, n] / x[m, ] / x[, n] / rownames(x) / colnames(x)

# ④ 행렬 연산자 #
# 전치행렬 : t(x) / 역행렬 : solve(x) / 행렬의 곱 : %*%

a <- matrix(c(1:4), nrow=2)
a
b <- matrix(c(1:4), nrow=2)
b
a+b
a*b
t(a)
solve(a)
a%*%b

### (5) 데이터 프레임Data Frame ###

# ② 데이터 프레임 문법 #
# data.frame(변수명1=벡터1, ...)
d <- data.frame(a=c(1, 2, 3, 4),
                b=c(2, 3, 4, 5),
                e=c('M', 'F','M', 'F'))
d
str(d)


### (6) 배열Array ###

# ② 배열 문법 #
# array(data, dim, dimnames)
rn = c('1행', '2행')
cn = c('1열', '2열')
mn = c('1차원', '2차원')
array(1:8,
      dim=c(2, 2, 2),
      dimnames=list(rn, cn, mn))


### (7) 팩터Factor ###
# 범주형 자료를 표현하기 위한 데이터 타입
# 범주형 데이터는 '순서형 데이터'와 '명목형 데이터'로 구분

# ② 팩터 문법
# factor(x, levels, labels, orederd)
factor('s',
       levels=c('s', 'l'))
factor(c('a', 'b', 'c'),
       ordered=TRUE)   # TRUE : 순서형, FALSE : 명목형 / 기본값은 FALSE



##### <7> 데이터 결합 #####


### (1) rbind ###
# 벡터, 행렬, 데이터 프레임의 '행'을 서로 결합
# rbind(데이터1, 데이터2, ...)
x <- data.frame(a=c("s1", "s2", "s3"),
                b=c("A", "B", "c"))
x
y <- data.frame(a=c("s5", "s6", "s7"),
                b=c("E", "F", "G"))
y
rbind(x, y)   # 열의 이름과 개수가 동일해야 함


### (2) cbind ###
# 벡터, 행렬, 데이터 프레임의 '열'을 서로 결합
# cbind(데이터1, 데이터2)
x <- data.frame(a=c("a", "b", "c"),
                b=c(80, 60, 70))
x
cbind(x, d=c(1, 2, 3))


### (3) merge
# 공통된 열을 하나 이상 가지고 있는 두 데이터 프레임에 대하여
# 기준이 되는 특정 컬럼의 값이 같은 행끼리 묶어 주는 병합 함수
# 데이터베이스에서 join과 동일한 역할
# merge(x, y, by, by.x, by.y, all=FALSE, all.x, all.y)
x <- data.frame(name=c("a", "b", "c"),
                math=c(1, 2, 3))
y <- data.frame(name=c("c", "b", "d"),
                english=c(4, 5, 6))
merge(x, y)
merge(x, y, all.x=TRUE)
merge(x, y, all=TRUE)



##### <8> 조건문 #####


### (1) if문 ###
score = 95
if( score>=90 ){
  print("수")
} else if( score>=80 ){
  print("우")
} else{
  print("가")
}


### (2) ifelse문 ###
# 조건식이 단순한 경우 참(TRUE)인 경우와 거짓(FALSE)인 경우로 경로를 분리해서 선택하는 명령문
# ifelse문은 벡터 연산이 가능

# ② ifelse 문법 #
# ifelse(조건식, 명령어1, 명령어2) : 조건식이 참이면 명령어1을 실행, 조건식이 거짓이면 명령어2를 실행
score = 95
ifelse( score>=60, "Pass", "Fail" )


### (3) switch문 ###
# 조건에 따라 여러 개의 경로 중 하나를 선택하여 명령어를 실행하는 명령문
course = "C"
switch(course,
       "A"="brunch",
       "B"="lunch",
       "dinner")



##### <9> 반복문 #####


### (1) for문 ###
for (i in 1:4) {
  print(i)
}


### (2) while문 ###
i=1
while (i<=4) {
  print(i)
  i=i+1
}


### (3) repeat문 ###
# 블록 안의 문장을 반복해서 수행하다가, 특정 상황에 종료할 때 사용하는 반복문
# repeat문은 탈출 조건이 없을 경우 무한하게 반복되기 때문에 탈출 조건을 블록 안에 명시해주어야 함
i=1
repeat {
  print(i)
  if (i>=2) {
    break
  }
  i=i+1
}


### (4) 루프 제어 명령어 ###

# ① break문 #
# 반복문을 중간에 탈출하기 위해 사용하는 명령어
for (i in 1:5) {
  print(i)
  if (i>=2) {
    break
  }
}

# ② next문 #
# 반복문에서 다음 반복으로 넘어갈 수 있도록 하는 명령어
for (i in 1:5) {
  if (i==2) {
    next
  }
  print(i)
}



##### <10> 사용자 정의 함수 #####

# ① 반환값이 있는 사용자 정의 함수 #
func_abs = function(x) {
  if (x<0) {
    return(-x)
  } else {
    return(x)
  }
}
func_abs(-10.9)
func_abs(10.1)

# ② 반환값이 없는 사용자 정의 함수 #
func_diff = function(x, y) {
  print(x)
  print(y)
  print(x-y)
}
val = func_diff(9, 1)
val





# ============================================================================ #





##### [2] 시각화 함수 #####





##### <1> graphics 패키지 #####

# ① plot 함수 #
# plot(x, y, 옵션)
a = c(3, 5, 4)
plot(a)

length <- iris$Petal.Length
plot(x=length)

length <- iris$Petal.Length
width <- iris$Petal.Width
plot(x=length,
     y=width)

length <- iris$Petal.Length
width <- iris$Petal.Width
plot(length,
     width,
     xlab="꽃잎 길이",
     ylab="꽃입 너비",
     main="꽃잎 길이와 너비")

# 꺾은선 그래프
x <- c(1, 5, 2, 4, 6, 9, 11, 8, 13)
plot(x,
     main="시계열 그래프 예제",
     type="b")                    # p, l, b, o, h, s, S, n / default : p

# ② hist 함수 #
# hist(x, 옵션)
length <- iris$Sepal.Length
hist(length,
     xlab="꽃받침 길이",
     ylab="수량",
     main="꽃받침 길이 분포")

length <- iris$Sepal.Length
hist(x=length,
     breaks=4,     # 계급 구간 : 4
     freq=FALSE)   # 상대도수

# ③ barplot 함수 #
# barplot(height, 옵션)
h <- c(15, 23, 5, 20)
name <- c("1분기", "2분기", "3분기", "4분기")
barplot(h,
        names.arg=name)   # names.arg : 각 막대에 사용할 문자열 벡터

h <- table(iris$Species)
h
barplot(h,
        ylab="수량",
        main="종별 수량")

# barplot(formula, data, 옵션)
# y~x로 y는 '수치형 데이터', x는 '범주형 데이터'
sales <- c(15, 23, 5, 20)
seasons <- c("1분기", "2분기", "3분기", "4분기")
df <- data.frame(sales,
                 seasons)
df
barplot(sales~seasons, data=df)
barplot(sales~seasons, df)

# ④ pie 함수 #
# pie(x, 옵션)
p <- c(15, 23, 5, 20)
l <- c("1분기", "2분기", "3분기", "4분기")
pie(x=p,
    labels=l)

p <- c(15, 23, 5, 20)
l <- c("1분기", "2분기", "3분기", "4분기")
pie(x=p,
    labels=l,
    density=10,     # 원그래프를 지정한 수만큼 사선을 그어서 표시
    angle=30*1:4)   # 첫 번째 파이 조각부터 30도씩 사선을 그을 각도를 지정

# ⑤ boxplot 함수 #
# boxplot(x, 옵션)
s <- iris$Sepal.Length
boxplot(s,
        main="꽃받침 길이 분포")

boxplot(iris$Sepal.Length~
          iris$Species,
        notch=TRUE,   # 중위수에 대한 신뢰구간을 움푹 들어간 형태로 표시
        xlab="종별",
        ylab="꽃받침 길이",
        main="종별 꽃받침 길이 분포")



##### <2> ggplot 패키지 #####
# 그래픽 문법(Grammar of Graphics)에 기반한 그래프(Plot)
# ggplot2를 처음 사용할 경우 ggplot 패키지를 설치해야 함
install.packages("ggplot2")
library(ggplot2)


### (2) ggplot 패키지의 구성요소 ###
# Data : 시각화하려는 데이터 또는 실제 정보를 의미
# Aesthetics : 축의 스케일, 색상, 채우기 등 미학적/시각적 속성을 의미
# Geometries(geoms) : 데이터를 표현하는 도형을 의미


### (3) ggplot 패키지의 주요 함수 ###

# ① geom_bar 함수 #
ggplot(diamonds,
       aes(color),
       )+geom_bar()

# ② geom_point 함수 #
ggplot(sleep,
       aes(x=ID,
           y=extra)
       )+geom_point()

# ③ geom_line 함수 #
ggplot(Orange,
       aes(age,
           circumference)
       )+geom_line(aes(color=Tree))

# ④ geom_boxplot 함수 #
ggplot(data=airquality,
       aes(x=Month,
           y=Temp,
           group=Month)
       )+geom_boxplot()
