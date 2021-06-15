use tip;

SELECT * FROM tip.tips;

SELECT total_bill, tip, smoker, time
FROM tips
LIMIT 5;

SELECT *, tip/total_bill as tip_rate
FROM tips
LIMIT 5;

SELECT *
FROM tips
WHERE time = 'Dinner'
LIMIT 5;

-- tips of more than $5.00 at Dinner meals
SELECT *
FROM tips
WHERE time = 'Dinner' AND tip > 5.00;

SELECT *
FROM tips
WHERE size >= 5 OR total_bill > 45;

SELECT sex, count(*)
FROM tips
GROUP BY sex;
/*
Female     87
Male      157
*/

SELECT day, AVG(tip), COUNT(*)
FROM tips
GROUP BY day;
/*
Fri   2.734737   19
Sat   2.993103   87
Sun   3.255132   76
Thur  2.771452   62
*/

SELECT smoker, day, COUNT(*), AVG(tip)
FROM tips
GROUP BY smoker, day;
/*
smoker day
No     Fri      4  2.812500
       Sat     45  3.102889
       Sun     57  3.167895
       Thur    45  2.673778
Yes    Fri     15  2.714000
       Sat     42  2.875476
       Sun     19  3.516842
       Thur    17  3.030000
*/

# 정렬
select day, avg(tip), count(*) from tips
group by day
order by day asc;

select day, avg(tip), count(*) from tips
group by day
order by day desc;