##### <1> big-O Notation #####


# n^2과 2^n 계산 결과 비교
for n in range(1, 15+1):
    print(n, n**2, 2**n)





# =========================================== #





##### <2> 자료형Data Type #####



### (1) 파이썬 자료형 ###

# * 숫자 * #
# object > int > bool
True == 1
False == 0


# * 매핑 * #
# 키와 자료형으로 구성된 복합 자료형


# * 집합 * #
# set은 중복된 값을 갖지 않는 자료형
a = set()
a
type(a)

a = {'a', 'b', 'c'}
type(a)
a = {'a': 'A', 'b': 'B', 'c': 'C'}
type(a)

a = {3, 2, 3, 5}   # set은 중복된 값이 있을 경우 하나의 값만 유지
a


# * 시퀀스 * #
a = 'abc'
a = 'def'
type(a)

a = 'abc'
id('abc')
id(a)
a = 'def'
id('def')
id(a)

# a[1] = 'd'   # 에러



### (2) 객체 ###


# 불변 객체 #
10
a = 10
b = a
id(10), id(a), id(b)


# 가변 객체 #
a = [1, 2, 3, 4, 5]
b = a
b
a[2] = 4
a
b


# (문법) is와 ==
a = [1, 2, 3]
a == a         # True
a == list(a)   # True
a is a         # True
a is list(a)   # False → list()로 한 번 더 묶어주면 별도의 객체로 복사되고 다른 ID를 갖게 됨

import copy
a = [1, 2, 3]
a == copy.deepcopy(a)   # True
a is copy.deepcopy(a)   # False → 값은 같지만 ID는 다르기 때문에
