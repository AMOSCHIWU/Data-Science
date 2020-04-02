##### 載入套件

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from feature_selector import FeatureSelector
```

##### 讀取檔案

```python
fixed_train_sr_2_4_loan=pd.read_csv("fixed_train_sr_2_4_loan.csv")
sr_3_51_features=pd.read_csv("sr_3_52_features.csv")
```

##### 將y1分隔出來並將資料要做標準化的欄位選出來

```python
y1=fixed_train_sr_2_4_loan.iloc[:,-1:]
sr_3_51_features=sr_3_51_features.drop(columns={'Unnamed: 0','y1','srno','YM'})
train_all= pd.concat([fixed_train_sr_2_4_loan,sr_3_51_features],axis = 1)
train_all=train_all.drop(columns={'Unnamed: 0','y1'})
ready=train_all.iloc[:,583:]
```

##### 對資料做標準化

```python
for column in ready.columns:
    ready[column] = (ready[column] - ready[column].mean()) / (ready[column].std())
```

##### 將標準化前原本的欄位丟掉再把標準化完的欄位以及y1合併

```python
train_all=train_all.drop(columns={'18', '27', '38', '48', '49', '51', '62', '68', '74', '76', '77', '81','97', '106', '121', '128', '146', '147', '160', '161', '189', '195','203', '207', '228', '238', '252', '257', '262', '263', '275', '277','290', '292', '300', '308', '314', '336', '340', '343', '350', '360','369', '383', '385', '409', '429', '431', '445', '448', '454'})
train_all= pd.concat([train_all,ready],axis = 1)
train_all= pd.concat([train_all,y1],axis = 1)
```

##### 將12月份的資料分割出來並對其餘的資料做特徵篩選

```python
train_all =train_all[(train_all["YM"] <201812)]
x_features = train_all.iloc[:,1:-1]
x_features = x_features.drop(columns=['YM'])
label_y1 = train_all.iloc[:,-1:]
fs = FeatureSelector(data = x_features, labels = label_y1)
fs.identify_collinear(correlation_threshold=0.8)
correlated_features = fs.ops['collinear']
fs.plot_collinear()
fs.identify_single_unique()
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
one_hot_features = fs.one_hot_features
base_features = fs.base_features
print('There are %d original features' % len(base_features))
print('There are %d one-hot features' % len(one_hot_features))
train = fs.data_all
zero_importance_features = fs.ops['zero_importance']
fs.plot_feature_importances(threshold = 0.99, plot_n = 39)
features_desc = fs.feature_importances
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
low_importance_features[:279]
```

##### 將不是特徵的欄位刪除

```python
train_all=train_all.drop(columns={'c_zip_514.0','c_zip_892.0','c_occp_403','c_zip_803.0','c_zip_263.0','c_zip_513.0','c_zip_630.0',
 'c_job_509','c_zip_724.0','c_occp_1008','c_occp_1007','c_occp_1018','c_zip_951.0','c_zip_249.0','c_zip_412.0','c_zip_305.0','c_zip_632.0','c_occp_410',
'c_occp_406','c_zip_923.0','c_zip_624.0','c_zip_269.0','c_zip_931.0','c_zip_826.0','c_zip_308.0','c_zip_325.0','c_zip_743.0','c_zip_337.0','c_zip_265.0','c_zip_643.0','c_zip_551.0','c_zip_400.0','c_zip_730.0','c_occp_502','c_zip_423.0','c_occp_806','c_zip_929.0','c_zip_438.0','c_zip_261.0','c_occp_808','c_zip_954.0','c_occp_903','c_zip_840.0','c_occp_103','c_zip_361.0','c_zip_828.0','c_zip_435.0','c_occp_303','c_zip_924.0','c_zip_648.0','c_occp_510','c_zip_913.0','c_occp_214','c_job_102','c_zip_266.0','c_zip_732.0','c_zip_734.0','c_zip_955.0',
'c_job_201','c_occp_819','c_zip_357.0','c_job_311','c_zip_557.0','c_zip_636.0','c_job_404','c_zip_846.0','c_job_503','c_zip_614.0','c_zip_881.0','c_occp_407','c_occp_820','c_zip_622.0','c_occp_507','c_zip_742.0','c_zip_369.0','c_job_301','c_job_305','c_job_307','c_job_310','c_job_501','c_job_507','c_zip_965.0','c_zip_252.0','c_occp_102','c_occp_202','c_occp_208','c_occp_509','c_zip_253.0','c_zip_267.0','c_occp_101','c_zip_982.0','c_job_502','c_zip_978.0','c_zip_546.0','c_zip_553.0','c_zip_555.0','c_zip_558.0','c_zip_602.0','c_zip_603.0','c_zip_979.0','c_zip_851.0','c_occp_704','c_occp_604','c_zip_972.0','c_zip_975.0','c_zip_976.0','c_zip_977.0','c_zip_883.0','c_occp_605','c_zip_884.0','c_occp_607','c_zip_940.0','c_zip_963.0','c_zip_941.0','c_zip_942.0','c_zip_943.0','c_zip_944.0','c_zip_926.0','c_zip_945.0','c_zip_952.0','c_zip_953.0','c_zip_956.0','c_zip_957.0','c_zip_958.0','c_zip_959.0','c_zip_947.0','c_zip_925.0','c_zip_922.0','c_zip_921.0','c_occp_701','c_occp_703','c_zip_966.0','c_zip_964.0','c_zip_530.0','c_zip_927.0','c_zip_885.0','c_zip_890.0','c_zip_891.0','c_zip_894.0','c_zip_901.0','c_zip_902.0','c_zip_905.0','c_zip_906.0','c_zip_911.0','c_occp_606','c_zip_527.0','c_zip_439.0','c_zip_524.0','c_zip_703.0','c_zip_615.0','c_zip_616.0','c_zip_631.0','c_zip_633.0','c_zip_634.0','c_occp_809','c_zip_635.0','c_zip_646.0','c_zip_647.0','c_zip_649.0','c_zip_653.0','c_zip_654.0','c_zip_655.0','c_zip_638.0','c_zip_713.0','c_zip_962.0','c_occp_1019','c_zip_512.0','c_zip_233.0','c_zip_212.0','c_zip_210.0','c_occp_1102','c_occp_1198','c_zip_228.0','c_occp_1015','c_zip_208.0','c_zip_209.0','c_zip_232.0','c_zip_226.0','c_occp_1020','c_zip_227.0','c_zip_211.0','c_zip_607.0','c_zip_715.0','c_zip_716.0','c_zip_352.0','c_zip_353.0','c_zip_354.0','c_zip_360.0','c_zip_362.0','c_zip_365.0',
'c_zip_336.0','c_zip_367.0','c_zip_424.0','c_zip_961.0','c_zip_605.0','c_zip_506.0','c_zip_509.0','c_zip_522.0','c_zip_368.0','c_zip_315.0','c_zip_313.0','c_zip_311.0','c_zip_272.0','c_zip_719.0','c_zip_725.0','c_zip_727.0','c_zip_733.0','c_zip_822.0','c_zip_823.0','c_zip_843.0','c_zip_844.0','c_zip_845.0','c_zip_847.0','c_zip_611.0','c_zip_606.0','c_zip_882.0','c_zip_426.0','c_zip_525.0','c_zip_909.0'})
```

##### 輸出檔案

```python
train_all.to_csv("sr_2_4_loan_sr_3_52_feature.csv")
```