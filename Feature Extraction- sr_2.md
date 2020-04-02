##### 載入套件

```python
import math
import pandas as pd
```

##### 讀取檔案

```python
sr_2 = pd.read_csv("sr_2.csv")
```

##### 對sr_2進行日期處理

```python
for index, date in enumerate(sr_2['txn_dt']):
    update = math.floor(date/100)
    sr_2['txn_dt'][index] = update
    print(index)
```

##### 將欄位丟掉並把正負值分開

```python
sr_2 = sr_2.drop(columns = {'Unnamed: 0','prod_type'})
sr_2 = sr_2[['srno', 'txn_dt', 'amt',]]
sr_3 = sr_2[sr_2['amt'] < 0]  
sr_4 = sr_2[sr_2['amt'] > 0]
```

##### 將正負值依照欄位作加總並合併且填補缺失值

```python
sr_3_group = sr_3.groupby(['srno','txn_dt']).sum().reset_index()    
sr_4_group = sr_4.groupby(['srno','txn_dt']).sum().reset_index()
sr_2last1= pd.merge(sr_3_group,sr_4_group,on = ['srno','txn_dt'],how = 'left')
sr_2last1=sr_2last1.fillna(0)
```

##### 輸出檔案

```python
sr_2last1.to_csv("sr_2_last1.csv")
```

