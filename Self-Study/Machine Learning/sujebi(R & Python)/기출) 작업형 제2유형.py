##### 작업형 제2유형 #####

print('========== 01 ==========')
# 데이터 가져오기
import pandas as pd
data = pd.read_csv('data/Train.csv')
print(data.head().T)

print('\n========== 02 ==========')
print(data.info())
print(data.isnull().sum())
print(data.isna().sum())

print('\n========== 03 ==========')
print(data.describe().T)

print('\n========== 04 ==========')
# 데이터 세트 분리
X = data.drop(columns=['Reached.on.Time_Y.N'])
y = data['Reached.on.Time_Y.N']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# id 저장
x_test_id = x_test['ID']
x_train = x_train.drop(columns=['ID'])
x_test = x_test.drop(columns=['ID'])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print('\n========== 05 ==========')
# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
print(x_train['Warehouse_block'].unique())
encoder = LabelEncoder()
x_train['Warehouse_block'] = encoder.fit_transform(x_train['Warehouse_block'])
print(x_train['Warehouse_block'].unique())
x_test['Warehouse_block'] = encoder.transform(x_test['Warehouse_block'])
print(x_test['Warehouse_block'].unique())

x_train['Mode_of_Shipment'] = encoder.fit_transform(x_train['Mode_of_Shipment'])
print(x_train['Mode_of_Shipment'].unique())
x_test['Mode_of_Shipment'] = encoder.transform(x_test['Mode_of_Shipment'])
print(x_test['Mode_of_Shipment'].unique())

x_train['Product_importance'] = encoder.fit_transform(x_train['Product_importance'])
print(x_train['Product_importance'].unique())
x_test['Product_importance'] = encoder.transform(x_test['Product_importance'])
print(x_test['Product_importance'].unique())

x_train['Gender'] = x_train['Gender'].replace('F', 0).replace('M', 1)
print(x_train['Gender'].unique())
x_test['Gender'] = x_test['Gender'].replace('F', 0).replace('M', 1)
print(x_test['Gender'].unique())

# # 원-핫 인코딩
# Warehouse_block_dummy = pd.get_dummies(x_train['Warehouse_block'], drop_first=True)
# print(Warehouse_block_dummy)
# x_train = pd.concat([x_train, Warehouse_block_dummy], axis=1)
# print(x_train.head().T)
# x_train = x_train.drop(columns=['Warehouse_block'])
# print(x_train.head().T)
# # 테스트 데이터에도 적용
# Warehouse_block_dummy = pd.get_dummies(x_test['Warehouse_block'], drop_first=True)
# x_test = pd.concat([x_test, Warehouse_block_dummy], axis=1)
# x_test = x_test.drop(columns=['Warehouse_block'])

print('\n========== 06 ==========')
# 표준화
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
print(x_train.describe().T)
print(x_test.describe().T)

print('\n========== 07 ==========')
# 모델링
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
xgb = XGBClassifier(random_state=10, use_label_encoder=False, eval_metric='error')
log = LogisticRegression(random_state=10)
rf = RandomForestClassifier(random_state=10)
gbm = GradientBoostingClassifier(random_state=10)
svc = SVC(random_state=10, probability=True)

xgb.fit(x_train, y_train)
log.fit(x_train, y_train)
rf.fit(x_train, y_train)
gbm.fit(x_train, y_train)
svc.fit(x_train, y_train)

y_test_predicted_xgb = pd.DataFrame(xgb.predict_proba(x_test))
y_test_predicted_log = pd.DataFrame(log.predict_proba(x_test))
y_test_predicted_rf = pd.DataFrame(rf.predict_proba(x_test))
y_test_predicted_gbm = pd.DataFrame(gbm.predict_proba(x_test))
y_test_predicted_svc = pd.DataFrame(svc.predict_proba(x_test))

y_test_pred_xgb = pd.DataFrame(xgb.predict(x_test))
y_test_pred_log = pd.DataFrame(log.predict(x_test))
y_test_pred_rf = pd.DataFrame(rf.predict(x_test))
y_test_pred_gbm = pd.DataFrame(gbm.predict(x_test))
y_test_pred_svc = pd.DataFrame(svc.predict(x_test))

print(y_test_predicted_gbm[1])

# 평가
from sklearn.metrics import roc_auc_score
print('### ROC-AUC ###')
print('XGBClassifier              :', roc_auc_score(y_test, y_test_predicted_xgb[1]))
print('LogisticRegression         :', roc_auc_score(y_test, y_test_predicted_log[1]))
print('RandomForestClassifier     :', roc_auc_score(y_test, y_test_predicted_rf[1]))
print('GradientBoostingClassifier :', roc_auc_score(y_test, y_test_predicted_gbm[1]))
print('SVC                        :', roc_auc_score(y_test, y_test_predicted_svc[1]))

from sklearn.metrics import accuracy_score
print('### Accuracy ###')
print('XGBClassifier              :', accuracy_score(y_test, y_test_pred_xgb))
print('LogisticRegression         :', accuracy_score(y_test, y_test_pred_log))
print('RandomForestClassifier     :', accuracy_score(y_test, y_test_pred_rf))
print('GradientBoostingClassifier :', accuracy_score(y_test, y_test_pred_gbm))
print('SVC                        :', accuracy_score(y_test, y_test_pred_svc))

from sklearn.metrics import r2_score
print('### R^2 ###')
print('XGBClassifier              :', r2_score(y_test, y_test_predicted_xgb[1]))
print('LogisticRegression         :', r2_score(y_test, y_test_predicted_log[1]))
print('RandomForestClassifier     :', r2_score(y_test, y_test_predicted_rf[1]))
print('GradientBoostingClassifier :', r2_score(y_test, y_test_predicted_gbm[1]))
print('SVC                        :', r2_score(y_test, y_test_predicted_svc[1]))

print('\n========== 08 ==========')
# 제출
final = pd.concat([x_test_id.reset_index(drop=True), y_test_predicted_gbm[1]], axis=1).rename(columns={1: 'Reached.on.Time_Y.N_proba'})
print(final)
final.to_csv('data/on_time.csv', index=False)

# 확인
final = pd.read_csv('data/on_time.csv')
print(final)