# Chapter 12 [회귀분석]





## 12.1 단순회귀모형



**회귀분석(regression analysis)** : 인과관계가 의심되는 복수의 변수를 사용하여 어느 변수로부터 다른 변수의 값을 예측하는 기법

**독립변수**와 **종속변수**



**단순회귀모형** : 회귀분석에서 설명변수와 반응변수가 1개씩인 가장 단순한 모델



**오차항** : 추정할 때 예측할 수 없는 부분
$$
Y_i=\beta_0+\beta_1x_i+e_i   (i=1, 2, ..., n)
$$
​			오차항끼리는 서로 독립이고,
$$
N(0, \sigma^2)
$$
​			즉 정규분포를 따른다.



### 단순 선형회귀 모형에 관한 가정

---

SR1. x의 각 값에 대해 y값은 다음과 같다.
$$
y = \beta_1+\beta_2x+e
$$


SR2. 무작위 오차 e의 기댓값은 다음과 같다.
$$
E(e)=0
$$
​		왜냐하면 다음과 같이 가정하였기 때문이다.
$$
E(y)=\beta_1+\beta_2x
$$


SR3. 무작위 오차 e의 분산은 다음과 같다.
$$
var(e)=\sigma^2=var(y)
$$
​		확률변수 y 및 e는 동일한 분산을 갖는다. 왜냐하면 이들은 단지 일정한 상수만큼 차이가 나기 때문이다.



SR4. 무작위 오차의 한 쌍인 e_i와 e_j의 공분산은 다음과 같다.
$$
cov(e_i, e_j)=cov(y_i, y_j)=0
$$
​		무작위 오차 e가 통계적으로 독립적인 경우 종속변수 y의 값도 통계적으로 독립적이라고 할 경우 더욱 강한 가정이 된다.



SR5. 변수 x는 확률적이지 않으며 최소한 2개의 상이한 값을 가져야 한다.

---



### 최소제곱법(ordinary least squares)

**최소제곱 잔차(least squares residuals)** : 각 관측값으로부터 적합하게 그은 선까지의 수직거리
$$
\hat e_i=y_i-\hat{y_i}=y_i-\hat\beta_1-\hat\beta_2x_i
$$


**최소제곱 잔차를 제곱한 합** 
$$
SSE=\sum_{i=1}^N\hat e^2
$$


**최소제곱 추정량**
$$
\hat\beta_2=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}
$$

$$
\hat\beta_1 = \bar y-\hat\beta_2\bar x
$$


https://hyerios.tistory.com/128



### 가우스-마코프 정리

선형회귀 모형에 관한 가정 SR1-SR5 하에서 추정량 `\hat\beta_1` 및 `\hat\beta_2`는 `\beta_1` 및 `\beta_2`의 모든 선형 및 불편 추정량 중에서 **최소의 분산**을 가지며, `\beta_1` 및 `\beta_2`의 최우수 불편 추정량(BLUE : Best Linear Unbiased Estimators)이다.



https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=jieun0441&logNo=221123143737





### t 검정

회귀계수에 대한 가설검정



검정통계량은
$$
t=\frac{\hat\beta_1-\beta_1}{\sqrt{\sigma^2C_1}}
$$
일반적으로 귀무가설은 `\beta_1=0`





## 12.2 중회귀모형

**중회귀모형** : 설명변수가 2개 이상인 모형





### 더미변수(dummy variable, 가변수)

**더미변수** : 질적변수를 변환하여 양적변수와 동일하게 취급할 수 있게 하는 기법



더미변수는 0과 1을 취하는 2진 변수로, 변환하고 싶은 질적변수의 카테고리 수에서 하나를 줄인 수만큼 필요(p.358)



https://kkokkilkon.tistory.com/37





## 12.3 모형의 선택





### 결정계수(R-squared)



- 총변동(**SST**) : 관측값 `y_i`가 어느 정도 분산되어 있는지 나타내는 지표
  $$
  \sum_{i=1}^n(y_i-\bar y)^2
  $$
  

- 회귀변동(**SSR**) : 예측값 `\hat y_i`가 관측값의 평균값 `\bar y`에 대해서 어느 정도 분산되어 있는지를 나타내는 지표

  
  $$
  \sum_{i=1}^n(\hat y_i-\bar y)^2
  $$

- 잔차변동(**SSE**) : 잔차의 산포도를 나타내는 지표
  $$
  \sum_{i=1}^n \hat e^2
  $$
  

***총변동 = 회귀변동 + 잔차변동***

즉, ***SST = SSR + SSE***


$$
R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}
$$




### 조정결정계수(adjusted R-squared)

**조정결정계수** : 설명변수를 추가했을 때 그 설명변수에 어느 정도 이상의 설명력이 없는 경우 결정계수의 값이 증가하지 않도록 조정하는 결정계수



SSE와 SST를 각각 자유도로 나누어 계산
$$
\bar R^2=1-\frac{SSE/(n-p-1)}{SST/(n-1)}
$$


***총변동의 자유도 = 회귀변동의 자유도 + 잔차변동의 자유도***

즉, ***n-1 = p + (n-p-1)***





### F 검정

(교재 가설설정 부분 i로 수정)

모형 전체에 대해서 수행(t 검정은 회귀계수에 대해서 수행)


$$
F=\frac{SSR/p}{SSE/(n-p-1)}
$$
조정결정계수와 같이 각각의 자유도로 나눠준다.



https://ko.wikipedia.org/wiki/%EB%B6%84%EC%82%B0_%EB%B6%84%EC%84%9D





### 최대로그우도와 AIC



- 우도
  $$
  L=\Pi_{i=1}f(x_i)
  $$

- 로그우도
  $$
  log L=\sum_{i=1}log f(x_i)
  $$

- 최대로그우도
  $$
  N(\hat y, \frac{1}{n}\sum_{i=1}^{n}\hat e^2)
  $$
  ​	의 밀도함수를 f(x)로 하여
  $$
  \sum_{i=1}^nlogf(y_i)
  $$
  **최대로그우도는 값이 클수록 모형의 적합도가 높다.**

- AIC

  최대로그우도의 문제점을 보완

  ***AIC = -2 × 최대로그우도 + 2 × 회귀계수의 수***

  ​	최대로그우도에 회귀계수의 수를 페널티로 부가(라쏘, 릿지와 비슷?)

- BIC

  ***BIC = -2 × 최대로그우도 + log n × 회귀계수의 수***

  **AIC와 BIC는 값이 작을수록 모형의 적합도가 높다.**   →   최대로그우도에 음수를 곱하기 때문에



https://rk1993.tistory.com/entry/AIC-BIC-Mallows-Cp-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0







