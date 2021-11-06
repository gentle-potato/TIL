### ★List Comprehension ###

# 파이썬은 map, filter와 같은 함수형(Functional) 기능을 지원하며, 람다 표현식(Lambda Expression)도 지원
a = list(map(lambda x: x + 10, [1, 2, 3]))
print(a)

b = [n * 2 for n in range(1, 10 + 1) if n % 2 == 1]
print(b)



### Generator ###

def get_natural_number():
    n = 0
    while True:
        n += 1
        yield n

g = get_natural_number()
print(g)
for _ in range(0, 100):
    print(next(g))

#
def generator():
    yield 1
    yield 'string'
    yield True

g = generator()
print(g)
print(next(g))
print(next(g))
print(next(g))



### range ###
a = [n for n in range(1000000)]
b = range(1000000)
print(len(a))
print(len(b))
# a에는 이미 생성된 값이 담겨 있고, b는 생성해야 한다는 조건만 존재

# 메모리 점유율 비교
import sys
print(sys.getsizeof(a))
print(sys.getsizeof(b))