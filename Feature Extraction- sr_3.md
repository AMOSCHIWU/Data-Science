##### 載入套件

```python
import math
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import feature_selector 
```

##### 讀檔

```python
sr_3=pd.read_csv("sr_3.csv")
sr_3 = sr_3.drop(columns = 'Unnamed: 0')
```

##### 將sr_3轉置

```python
sr_3=sr_3.pivot_table('TF',['srno','DATA_DATE'],'TERM')
```

##### 填補缺失值

```python
sr_3= sr_3.fillna(0)
```

##### 留下關鍵字並移除顧客編號及日期

```python
sr_3_TERM_feature = sr_3.iloc[:,2:]
```

##### 將出現次數小於2000的刪除

```python
sum_frequency = []
for freq in sr_3_TERM_feature.columns:
    sum_frequency.append(sr_3_TERM_feature[freq].sum())
col = []
for column in sr_3_TERM_feature.columns:
    col.append(column)
combine = []
for index, name in enumerate(col):
    combine.append([name,sum_frequency[index]])
delete = []
for component in combine:
    if component[1] < 2000:
        delete.append(component[0])
know = []
for component in combine:
    if component[1] > 2000:
        know.append([component[0],component[1]])
know = pd.DataFrame(know)
sr_3 = sr_3.drop(columns = delete)
```

##### 賦予每個欄位數值名稱

```python
TERM = sr_3.iloc[:,2:]
col_rename = []
for i in range(465):
    col_rename.append(i)
TERM.columns = col_rename
```

##### 將顧客編號與年份合併進來

```python
srno_YM = sr_3.iloc[:,0:2]
sr_3_TERM = pd.concat([srno_YM,TERM],axis=1)
```

##### 讀取before_encoding_training_data檔案

```python
train = pd.read_csv('before_encoding_training_data.csv')
train = train.drop(columns = 'Unnamed: 0')
```

##### 將顧客編號與年份合併

```python
srno = train['srno']
YM = train['YM']
srno_YM = pd.concat([srno,YM],axis = 1)
```

##### 將DATA_DATE重新命名為YM

```PYTHON
sr_3_TERM = sr_3_TERM.rename(columns = {'DATA_DATE':'YM'})
```

##### 合併srno_YM和sr_3_TERM並填補缺失值

```PYTHON
srno_YM_sr_3 = pd.merge(srno_YM,sr_3_TERM,on = ['srno','YM'],how = 'left')
srno_YM_sr_3 = srno_YM_sr_3.fillna(0)
```

##### 讀取y1的檔案並做日期處理

```python
y1 = pd.read_csv('result_y1.csv')
for index,date in enumerate(y1['YYYYMM']):
    update = date - 1
    y1['YYYYMM'][index] = update

y1['y1'] = 1

y1 = y1.rename(columns = {'YYYYMM':'YM'})
```

##### 將srno_YM_sr_3和y1合併並填補缺失值

```python
srno_YM_sr_3_loan = pd.merge(srno_YM_sr_3,y1,on = ['srno','YM'],how = 'left')
srno_YM_sr_3_loan = srno_YM_sr_3.fillna(0)
```

##### 移除y1

```python
srno_YM_sr_3 = srno_YM_sr_3.iloc[:,:-1]
```

##### 讀取y2的檔案並做日期處理

```python
y2 = pd.read_csv('result_y2.csv')
y2 = y2.rename(columns = {'YYYYMM':'YM'})
y2['y2'] = 1

for index,date in enumerate(y2['YM']):
    update = date - 1
    y2['YM'][index] = update
```

##### 合併y2的檔案並填補缺失值

```python
srno_YM_sr_3_fund = pd.merge(srno_YM_sr_3,y2,on = ['srno','YM'],how = 'left')
srno_YM_sr_3_fund = srno_YM_sr_3_fund.fillna(0)
```

##### 將小於12月y1的資料選出來並做標準化

```python
fit_feature_selector = srno_YM_sr_3_loan[srno_YM_sr_3_loan['YM'] < 201812]

fit_feature = fit_feature_selector.iloc[:,2:-1]
fit_label  = fit_feature_selector.iloc[:,-1:]

for column in fit_feature.columns:
    fit_feature[column] = (fit_feature[column] - fit_feature[column].mean()) / (fit_feature[column].std())
```

##### 做y1特徵篩選

```python
fs = FeatureSelector(data = fit_feature, labels = fit_label)
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)

zero_importance_features = fs.ops['zero_importance']
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']

im = fs.feature_importances

use = []

for index, nom in enumerate(im['normalized_importance']):
    if nom < 0.00507052:
        use.append(im['feature'][index])

srno_YM_sr_3_loan = srno_YM_sr_3_loan.drop(columns = use)
```

##### 將小於12月y2的資料選出來並做標準化

```python
fit_feature_selector2 = srno_YM_sr_3_fund[srno_YM_sr_3_fund['YM'] < 201812]

fit_feature = fit_feature_selector2.iloc[:,2:-1]
fit_label  = fit_feature_selector2.iloc[:,-1:]

for column in fit_feature.columns:
    fit_feature[column] = (fit_feature[column] - fit_feature[column].mean()) / (fit_feature[column].std())
```

##### 做y2特徵篩選

```python
fs = FeatureSelector(data = fit_feature, labels = fit_label)
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)

zero_importance_features = fs.ops['zero_importance']
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']

im = fs.feature_importances

use2 = []

for index, nom in enumerate(im['normalized_importance']):
    if nom < 0.00506931:
        use2.append(im['feature'][index])

srno_YM_sr_3_fund = srno_YM_sr_3_fund.drop(columns = use2)
```

##### 輸出檔案

```python
srno_YM_sr_3_fund.to_csv('sr_3_46_features_fund.csv')
srno_YM_sr_3_loan.to_csv('sr_3_52_features.csv')
```

