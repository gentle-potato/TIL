06/08



## Pandas에서의 Indexing & Slicing

`df`라는 딕셔너리가 있다고 할 때,

예를 들어 `df[0:2]`는 0, 1의 index만 불러오고 경계값(2)은 불러오지 않는다.



### `iloc[]`과 `loc[]`

'location'을 의미한다.

`iloc`은 index로 찾고, `loc`은 라벨값으로 찾는다.





## 데이터 해석

데이터 분포가 한 쪽으로 치우쳤다면(***포아송분포***),

정상적으로 Pearson 상관계수를 확인하려면 `log`를 사용해야 한다.

이렇게 해서 **선형관계(linear)**를 만들어야 상관관계가 확인된다!



또한 다른 스케일의 숫자형 데이터나 범주형 데이터는

 `log`를 통해 같은 스케일(scale)로 맞춰줘야 데이터 손실을 예방할 수 있다.

→ 이렇게 숫자형 데이터를 encoding 하면,

​	 pandas-profiling을 했을 때 원래는 나타나지 않았던 변수들에 대한 상관관계도 반환한다.

#### Before encoding

![(0608)_01. before encoding](0608_abstraction.assets/(0608)_01. before encoding-1623165157933.PNG)

#### After encoding

![(0608)_02. after encoding](0608_abstraction.assets/(0608)_02. after encoding.PNG)



<**추가로 설치한 `pip`**>

```python
pip install plotly
pip install scikit-learn==0.23.2
```





## 프로젝트 안내

1. 주제를 정해 데이터를 수집해서,
2. pre-processing해서 그림으로 잘 보여주고,
3. 상관관계를 분석해서 ***"스토리"***를 구성!

++ 머신러닝
