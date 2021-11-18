##### <1> 리스트List #####



### (1) 리스트의 활용 방법 ###

a = [1, 2, 3]
a
a.append(4)
a
a.insert(3, 5)   # 3번째 인덱스에 5를 삽입
a
a[1:4:2]   # 세 번째 파라미터는 단계(Step)의 의미

# a[9]   # 존재하지 않는 인덱스를 조회할 경우 IndexError 발생
try:
    print(a[9])
except IndexError:
    print("존재하지 않는 인덱스")

# 리스트에서 요소 삭제하기
# 1)) del : 인덱스의 위치에 있는 요소 삭제
a = [1, 2, 3, 5, 4, '안녕', True]
a
del a[1]
a
# 2)) remove : 값에 해당하는 요소 삭제
a
a.remove(3)
a
# 3)) pop : 추출로 처리되며, 삭제될 값을 리턴하고 삭제가 진행
a
a.pop(3)
a



### (2) 리스트의 특징 ###
# 리스트는 객체로 되어 있는 모든 자료형을 '포인터'로 연결





##### <2> 딕셔너리Dictionary #####



### (1) 딕셔너리의 활용 방법 ###

a = {'key1': 'value1', 'key2': 'value2'}
a
a['key3'] = 'value3'
a

a['key1']
# a['key4']   # 존재하지 않는 키를 조회할 경우 KeyError 발생
try:
    print(a['key4'])
except KeyError:
    print('존재하지 않는 키')

# 딕셔너리에서 키 삭제하기
# del a['key4']   # KeyError

'key4' in a
if 'key4' in a:
    print('존재하는 키')
else:
    print('존재하지 않는 키')

# 키/값 꺼내오기
for k, v in a.items():
    print(k, v)

del a['key1']
a



### (2) 딕셔너리 모둘 ###


# defaultdict 객체 #
# 존재하지 않는 키를 조회할 경우, 에러 메시지를 출력하는 대신 디폴트값을 기준으로 해당 키에 대한 딕셔너리 아이템을 생성
import collections
a = collections.defaultdict(int)
a['A'] = 5
a['B'] = 4
a
a['C'] += 1
a


# Counter 객체 #
# 아이템에 대한 개수를 계산해 딕셔너리로 반환
a = [1, 2, 3, 4, 5, 5, 5, 6, 6]
b = collections.Counter(a)
b
type(b)

b.most_common(2)


# OrderdDict 객체 #
# 입력 그대로 순서가 유지
collections.OrderedDict({'banana': 3, 'apple': 4, 'pear': 1, 'orange': 2})