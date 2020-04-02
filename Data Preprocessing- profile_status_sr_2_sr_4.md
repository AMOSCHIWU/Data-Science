##### 載入套件

```python
import math
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
```

##### 讀取檔案

```python
train = pd.read_csv('before_encoding_training_data.csv')
sr_4_fund = pd.read_csv('sr_4_fund.csv')
sr_2 = pd.read_csv('sr_2_last1.csv')
sr_4_loan = pd.read_csv('sr_4_loan.csv')
train = train.drop(columns = 'Unnamed: 0')
sr_4_fund= sr_4.drop(columns = 'Unnamed: 0')
sr_2 = sr_2.drop(columns = 'Unnamed: 0')
```

#####  將不必要欄位移除

```python
sr_4_loan = sr_4_loan.drop(columns = 'Unnamed: 0')
sr_4_fund= sr_4_fund.drop(columns = 'level_1')
sr_4_loan = sr_4_loan.drop(columns = 'level_1')
```

##### 依據欄位加總score

```python
sr_4_fund= sr_4_fund.groupby(['level_0','level_2']).sum().reset_index()
sr_4_loan = sr_4_loan.groupby(['level_0','level_2']).sum().reset_index()
```

##### 將欄位重新命名

```python
sr_4_fund=sr_4_fund.rename(columns= {'level_0': 'srno','level_2':'YM'})
sr_2 = sr_2.rename(columns = {'txn_dt': 'YM','amt_x': 'amt2', 'amt_y' : 'amt3'})
sr_4_loan = sr_4_loan.rename(columns= {'level_0': 'srno','level_2':'YM'})
```

##### 將train的數值欄位做標準化

```python
standardize_col = ['n_age', 'py_a', 'py_b', 'py_c', 'py_d', 'py_e',
       'py_f', 'py_g', 'py_h', 'py_i', 'py_j', 'py_k', 'py_l', 'as_a', 'as_b',
       'as_c', 'as_d', 'as_e', 'as_f', 'as_g', 'as_h', 'as_i', 'as_j', 'as_k',
       'as_l', 'as_m', 'as_n', 'scr', 'amt']
	for col in standardize_col:
train[col] = (train[col] - train[col].mean()) / (train[col].std())

```


##### 將非數值欄位做獨熱編碼

```python
one_hot_col = ['c_gender','c_mry','c_zip', 'c_edu', 'c_job', 'c_occp', 'x_flg_house', 'CAR_FLG','a_incm_flg']

one_hot_c_gender = pd.get_dummies(train['c_gender'],prefix = 'c_gender')
one_hot_c_mry = pd.get_dummies(train['c_mry'],prefix = 'c_mry')
one_hot_c_zip = pd.get_dummies(train['c_zip'],prefix = 'c_zip')
one_hot_c_edu = pd.get_dummies(train['c_edu'],prefix = 'c_edu')
one_hot_c_job = pd.get_dummies(train['c_job'],prefix = 'c_job')
one_hot_c_occp = pd.get_dummies(train['c_occp'],prefix = 'c_occp')
one_hot_x_flg_house = pd.get_dummies(train['x_flg_house'],prefix = 'x_flg_house')
one_hot_CAR_FLG = pd.get_dummies(train['CAR_FLG'],prefix = 'CAR_FLG')
one_hot_a_incm_flg = pd.get_dummies(train['a_incm_flg'],prefix = 'a_incm_flg')
```

##### 丟掉欄位並將欄位重新排序

```python
train = train.drop(columns = one_hot_col )

train = train[['srno', 'YM', 'n_age', 'py_a', 'py_b', 'py_c', 'py_d', 'py_e', 'py_f','py_g', 'py_h', 'py_i', 'py_j', 'py_k', 'py_l', 'as_a', 'as_b', 'as_c',
'as_d', 'as_e', 'as_f', 'as_g', 'as_h', 'as_i', 'as_j', 'as_k', 'as_l','as_m', 'as_n', 'scr', 'amt', 'y1']]
```

##### 選出欄位並將其合併

```python
srno_YM_age = train.iloc[:,0:3]
middle_status = train.iloc[:,3:31]
y1 = train.iloc[:,-1:]
train = pd.concat([srno_YM_age,one_hot_c_gender],axis = 1)
train = pd.concat([train,one_hot_c_mry],axis = 1)
train = pd.concat([train,one_hot_c_zip],axis = 1)
train = pd.concat([train,one_hot_c_edu],axis = 1)
train = pd.concat([train,one_hot_c_job],axis = 1)
train = pd.concat([train,one_hot_c_occp],axis = 1)
train = pd.concat([train,one_hot_a_incm_flg],axis = 1)
train = pd.concat([train,one_hot_x_flg_house],axis = 1)
train = pd.concat([train,one_hot_CAR_FLG],axis = 1)
train = pd.concat([train,middle_status],axis = 1)
train = pd.concat([train,y1],axis = 1)
```

##### 讀入y1跟y2的檔案

```python
#%%
y1 = pd.read_csv('result_y1.csv')
y1 = y1.rename(columns = {'YYYYMM':'YM'})
y1['y1'] = 1
for index,date in enumerate(y1['YM']):
    update = date - 1
    y1['YYYYMM'][index] = update
y2 = pd.read_csv('result_y2.csv')
y2 = y2.rename(columns = {'YYYYMM':'YM'})
y2['y2'] = 1

for index,date in enumerate(y2['YM']):
    update = date - 1
    y2['YM'][index] = update
```

##### 將資料合併

```python

train = train.iloc[:,:-1]

train_sr_2 = pd.merge(train,sr_2,on=['srno','YM'],how = 'left')

train_sr_2_sr_4_loan = pd.merge(train_sr_2,sr_4_loan,on=['srno','YM'],how = 'left')

train_sr_2_sr_4_loan = pd.merge(train_sr_2_sr_4_loan,y1,on = ['srno','YM'],how = 'left')
train_sr_2_sr_4_fund = pd.merge(train_sr_2,sr_4_fund,on=['srno','YM'],how = 'left')
train_sr_2_sr_4_fund = pd.merge(train_sr_2_sr_4_fund,y2,on = ['srno','YM'],how = 'left')

```

##### 填補缺失值

```python

train_sr_2_sr_4_loan = train_sr_2_sr_4_loan.fillna(0)

train_sr_2_sr_4_fund = train_sr_2_sr_4_fund.fillna(0)
```

#####  修改錯誤 先做合併且填補缺失值

```python
train_fixed = train_sr_2_sr_4_loan.iloc[:,:-4]
train_fixed2 =  train_sr_2_sr_4_fund.iloc[:,:-4]
y1 = train_sr_2_sr_4_loan.iloc[:,-1:]
y2 = train_sr_2_sr_4_fund.iloc[:,-1:]
train_fixed = pd.merge(train_fixed,sr_2,on=['srno','YM'],how = 'left')
train_fixed = pd.merge(train_fixed,sr_4,on=['srno','YM'],how = 'left')
```

##### 修改錯誤 再將amt2,amt3,score欄位標準化

```python
standardize_col = ['amt2','amt3','score']
	for col in standardize_col:
    	train_fixed[col] = (train_fixed[col] - train_fixed[col].mean()) / (train_fixed[col].std())
```

##### 做最後的合併

```
train_fixed_finish = pd.concat([train_fixed,y2],axis=1)
train_fixed_finish2 = pd.concat([train_fixed,y1],axis=1)
```

##### 輸出檔案

```python
train_fixed_finish.to_csv('fixed_train_sr_2_4_fund.csv')
train_fixed_finish2.to_csv('fixed_train_sr_2_4_loan.csv')
```

