# MySQL과 Pandas에서 SQL 사용하기



## 데이터와 스키마 만들기



### 데이터

`SELECT`와 `WHERE` `JOIN`에 관한 내용은 알고 있으니, 실제로 작성해본 코드만 기록한다.

```mysql
# 사용할 스키마를 선언해주는 것이 편하다.
use classicmodels;
show tables;

# 어제 배운 내용 복습
select * from employees;
select firstName, lastName from employees;

select employeeNumber, firstName, lastName from employees
where employees.employeeNumber >= 1300;

select * from offices;
select city, phone from offices where offices.officeCode = '1';

# LEFT JOIN
select customers.customerNumber, orders.orderNumber, customers.country from orders
left join customers
on customers.customerNumber = orders.customerNumber;

select customers.customerNumber, payments.checkNumber from customers
left join payments
on customers.customerNumber = payments.customerNumber;

# INNER JOIN
select orders.customerNumber, orders.orderNumber, customers.country from orders
inner join customers
on orders.customerNumber = customers.customerNumber
where customers.country = 'USA';

# 별칭 사용
select A.customerNumber, A.orderNumber, B.country from orders A
inner join customers B
on A.customerNumber = B.customerNumber
where B.country = 'USA';

# 내보낼 csv 파일 - customers 테이블에 없는 checkNumber 컬럼 추가하기
select customers.state, customers.customerName, payments.checkNumber from customers
left join payments
on customers.customerNumber = payments.customerNumber
where payments.paymentDate >= '2004-10-06';
```



`LEFT JOIN`과 `INNER JOIN`만 알고있으면 된다. (`RIGHT JOIN`은 `Null`이 많이 나온다...)



### 스키마

교재 p.67 참고

