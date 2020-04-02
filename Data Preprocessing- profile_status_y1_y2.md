##### 載入套件

```python
import pandas as pd
import numpy as np
import math
```

##### 讀取檔案

```python
profile = pd.read_csv("profile.csv")
status = pd.read_csv("status.csv")
sr_1 = pd.read_csv("sr_1.csv", encoding = 'gb18030')
sr_1 = sr_1.drop(columns = ['EFF_DT','TXN_DESC','mcc'])
y1 = pd.read_csv("result_y1.csv")
y2 = pd.read_csv("result_y2.csv")
```

##### 將sr_1相同帳單日期的金額先合併

```python
STMT_DT = np.array(sr_1['STMT_DT'])
srno = np.array(sr_1['srno'])
amt = np.array(sr_1['amt'])
sr_1 = sr_1['amt'].groupby([srno,STMT_DT]).sum().to_frame('amt').reset_index()
sr_1.columns = ["srno","YM","amt"]
```

##### 將sr_1的日期進行處理

```python
#%%
sr_1['re'] = 0
for index,date in enumerate(sr_1['YM']):
    sr_1['re'][index] = math.floor(date/100)
sr_1.drop(columns=['YM'])
sr_1 = sr_1[['srno','re','amt']]
sr_1.columns = ["srno","YM","amt"]
```

##### 將sr_1的amt以srno跟YM進行合併

```python
YM = np.array(sr_1['YM'])
srno = np.array(sr_1['srno'])
amt = np.array(sr_1['amt'])
sr_1 = sr_1['amt'].groupby([srno,YM]).sum().to_frame('amt').reset_index()
sr_1.columns = ["srno","YM","amt"]
```

##### 將profile跟status檔案進行合併

```python
profile_status = pd.merge(profile,status,how = 'left',on = 'srno')
profile_status = profile_status.rename(columns={'YYYYMM':'YM'})
```

##### 將合併完的檔案合併sr_1檔案並填補NAN值

```python
profile_status_sr_1 = pd.merge(profile_status,sr_1,on= ['srno','YM'],how = 'left')
profile_status_sr_1['amt'] = profile_status_sr_1['amt'].fillna(0)
```

##### 輸出合併完的檔案

```python
profile_status_sr_1.to_csv('profile_status_sr_1')
```

##### 將y1的月份先進行處理

```python
for index, yyyymm in enumerate(y1['YYYYMM']):
    y1['YYYYMM'][index] = yyyymm - 1
y1= y1.rename(columns = {'YYYYMM':'YM'})
y1['y1'] = 1
```

##### 將y1的資料合併profile_status_sr_1檔案

```python
profile_status_sr_1_y1 = pd.merge(profile_status_sr_1,y1,on= ['srno','YM'],how = 'left')
profile_status_sr_1_y1['y1'] = profile_status_sr_1_y1['y1'].fillna(0)
```

##### 輸出合併完的檔案

```python
profile_status_sr_1_y1.to_csv('before_encoding_training_data.csv')
```

##### 將y2的月份進行處理

```python
for index, yyyymm in enumerate(y2['YYYYMM']):
    y2['YYYYMM'][index] = yyyymm - 1
y2= y2.rename(columns = {'YYYYMM':'YM'})
y2['y2'] = 1
```

##### 將y2的資料合併profile_status_sr_1檔案

```python
profile_status_sr_1_y2 = pd.merge(profile_status_sr_1,y2,on= ['srno','YM'],how = 'left')
profile_status_sr_1_y2['y2'] = profile_status_sr_1_y2['y2'].fillna(0)
```

##### 輸出合併完檔案

```python
profile_status_sr_1_y2.to_csv('before_encoding_training_data2.csv')
```