06/07

# Pandas&Kaggle



## 환경 구축

PyPI.org에서 패키지 다운로드

>1. pandas 1.2.3
>2. pandas-profiling 2.12.0
>3. ipywidgets 7.6.3
>4. ploty 4.14.3
>5. scikit-learn 0.23.2

:heavy_check_mark:**프로젝트마다 맞는 환경을 구축하는 것이 가장 중요하다.**



## Pandas

**대부분이 딕셔너리 타입(key, value)으로 데이터 검색**

왜? structured data가 아닌 ***log data***로 바뀌는 추세!

- Series(딕셔너리) : Map/Reduce   ->   key와 value
- DataFrame(엑셀이라고 생각) : index와 column



### <`1_Pandas.ipynb`>

petal : 꽃잎, sepal : 꽃받침



**① 데이터 불러오기**

```python
url =
df_iris_sample = pd.read_csv(url)
```

![(0607)_01. 데이터 불러오기](0607_Pandas&Kaggle.assets/(0607)_01. 데이터 불러오기.PNG)

- `.info()` : 데이터 타입 확인
- `.describe()` : `min`, `max`, `,std` 등
- `.profile_report()` : pandas-profiling
- `.head()` : 데이터 앞쪽 표시
- `.tail()` : 데이터 뒤쪽 표시
- `.columns` : index 목록



**② pandas-profiling**

![(0607)_02. pandas-profiling_1](0607_Pandas&Kaggle.assets/(0607)_02. pandas-profiling_1.PNG)

![(0607)_03. pandas-profiling_2](0607_Pandas&Kaggle.assets/(0607)_03. pandas-profiling_2.PNG)

![(0607)_04. pandas-profiling_3](0607_Pandas&Kaggle.assets/(0607)_04. pandas-profiling_3.PNG)

- **파랑** : **양**의 상관관계
- **빨강** : **음**의 상관관계
- **짙을수록 상관관계 ↑**



![(0607)_05. pandas-profiling_4](0607_Pandas&Kaggle.assets/(0607)_05. pandas-profiling_4.PNG)

​	변수를 x축과 y축으로 설정해서 관계를 확인할 수 있다.

![(0607)_06. pandas-profiling_5(높은 상관관계)](0607_Pandas&Kaggle.assets/(0607)_06. pandas-profiling_5(높은 상관관계).PNG)

​	높은 상관관계를 가지는 경우 도표에 나타나지 않는다.



## Kaggle

혼자서 단계별로 코딩에 대해 공부할 수 있는 사이트

- url : https://www.kaggle.com/



- 데이터 읽어올 때 `index_col` 값에 따른 차이

  ① `index_col=None`

  ![(0607)_08. index_col=None](0607_Pandas&Kaggle.assets/(0607)_08. index_col=None-1623163805283.PNG)

  ​	`Unnamed`가 나온다.

  

  ② `index_col=0`

  ![(0607)_07. index_col=0](0607_Pandas&Kaggle.assets/(0607)_07. index_col=0.PNG)

  ​		`Unnamed`가 나오지 않는다.



- 기타 커맨드(***시험기간이라 불가피하게 사진으로...***)

![(0607)_11.](0607_Pandas&Kaggle.assets/(0607)_11.-1623163829488.jpg)



