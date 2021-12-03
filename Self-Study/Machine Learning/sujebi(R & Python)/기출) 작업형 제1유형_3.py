print('========== 01 ==========')
# 데이터 가져오기
import pandas as pd
data = pd.read_csv('data/Insurance.csv')
print(data.head())

print('\n========== 02 ==========')
avg = data['charges'].mean()
sd = data['charges'].std()
outlier_max = avg + 1.5*sd
print(data[data['charges']>=outlier_max]['charges'].sum())