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



### enumerate ###
a = ['a1', 'b2', 'c3']
for i, v in enumerate(a):
    print(i, v)



### locals ###
# locals()는 로컬 심볼 테이블 딕셔너리를 가져오는 메소드로 업데이트 또한 가능
import pprint
pprint.pprint(locals())
# pprint로 출력하게 되면 보기 좋게 줄바꿈 처리를 해주기 때문에 가독성 ↑



### 변수명과 주석 ###
# 어느 코드가 보기 좋은가?

# 1)
# def numMatchingSubseq(self, S: str, words: List[str]) -> int:
#     a = 0
#
#     for b in words:
#         c = 0
#         for i in range(len(b)):
#             d = S[c:].find(b[i])
#             if d < 0:
#                 a -= 1
#                 break
#             else:
#                 c += d + 1
#         a += 1
#
#     return a

# 2)
# def numMatchingSubseq(self, S: str, words: List[str]) -> int:
#     matched_count = 0
#
#     for word in words:
#         pos = 0
#         for i in range(len(word)):
#             # Find matching position for each character.
#             found_pos = S[pos:].find(word[i])
#             if found_pos < 0:
#                 matched_count -= 1
#                 break
#             else:   # If found, take step position forward.
#                 pos += found_pos + 1
#         matched_count += 1
#
#     return matched_count

# 당연히 2)!



### Zen of Python ###
import this
# The Zen of Python, by Tim Peters