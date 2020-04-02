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
fixed_train_sr_2_4_fund=pd.read_csv("fixed_train_sr_2_4_fund.csv")
sr_3_45_features=pd.read_csv("sr_3_46_features_fund.csv")
```

##### 將y2分隔出來並且資料要做標準化的欄位選出來

```python
fixed_train_sr_2_4_fund=pd.read_csv("fixed_train_sr_2_4_fund.csv")
sr_3_45_features=pd.read_csv("sr_3_45_features_fund.csv")
y2=fixed_train_sr_2_4_fund.iloc[:,-1:]
sr_3_45_features=sr_3_45_features.drop(columns={'Unnamed: 0','y2','srno','YM'})
train_all= pd.concat([fixed_train_sr_2_4_fund,sr_3_45_features],axis = 1)
train_all=train_all.drop(columns={'Unnamed: 0','y2'})
ready=train_all.iloc[:,583:]
```

##### 對資料做標準化

```python
for column in ready.columns:
    ready[column] = (ready[column] - ready[column].mean()) / (ready[column].std())
```

##### 將標準化前原本的欄位丟掉再把標準化完的欄位以及y2合併

```python
train_all=train_all.drop(columns={'0', '6', '27', '43', '48', '57', '74', '81', '86', '97', '106', '130','134', '142', '146', '147', '160', '161', '189', '207', '252', '257','272', '277', '290', '294', '308', '314', '325', '334', '342', '343','351', '360', '384', '385', '387', '398', '409', '429', '431', '436','445', '448', '454'})
train_all= pd.concat([train_all,ready],axis = 1)
train_all= pd.concat([train_all,y2],axis = 1)
```

##### 將12月份的分割出來並且對其餘的資料做特徵篩選

```python
train_all =train_all[(train_all["YM"] <201812)]
x_features = train_all.iloc[:,1:-1]
x_features = x_features.drop(columns=['YM'])
label_y1 = train_all.iloc[:,-1:]
fs = FeatureSelector(data = x_features, labels = label_y1)
print('go2')
fs.identify_collinear(correlation_threshold=0.8)
correlated_features = fs.ops['collinear']
fs.plot_collinear()
fs.identify_single_unique()
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
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
low_importance_features[:287]
```

##### 將不是特徵的欄位刪除

```python
train_all=train_all.drop(columns={'c_occp_1001','c_zip_269.0','c_zip_505.0','c_zip_540.0','c_zip_603.0','c_zip_881.0','c_mry_O','c_zip_338.0','c_zip_520.0','c_occp_805','c_occp_1198','c_job_101','c_zip_612.0','c_zip_711.0','c_occp_410','c_zip_721.0','c_zip_367.0','c_zip_247.0','c_zip_308.0','c_zip_307.0','c_job_599','c_zip_202.0','c_occp_1005','c_job_508','c_zip_803.0','c_occp_1014','c_zip_904.0','c_zip_558.0','334','c_zip_335.0','454','c_zip_326.0','c_zip_421.0','c_zip_423.0','c_occp_811','c_job_311','c_occp_1007','c_zip_306.0','c_zip_950.0','c_zip_265.0','c_zip_557.0','c_zip_249.0','c_occp_1105','c_zip_328.0','c_occp_814','c_occp_820','c_zip_723.0','c_occp_510','c_zip_911.0','c_zip_807.0','as_l','c_occp_1016','c_zip_260.0','c_zip_546.0','c_zip_512.0','c_zip_262.0','c_zip_503.0','c_zip_305.0','c_zip_608.0','387','342','c_occp_209','c_zip_360.0','c_occp_405','c_zip_622.0','c_zip_361.0','c_zip_414.0','c_zip_640.0','c_occp_801','c_zip_632.0','c_zip_602.0','c_occp_816','c_occp_818','c_occp_821','c_zip_422.0','c_occp_807','c_zip_823.0','c_job_409','c_job_408','c_occp_703','c_zip_827.0','c_occp_102','c_occp_103','c_job_507','c_zip_800.0','c_zip_824.0','c_zip_743.0','c_occp_205','c_zip_742.0','c_zip_717.0','c_zip_736.0','c_zip_735.0','c_zip_718.0','c_zip_734.0','c_zip_733.0','c_zip_727.0','c_zip_726.0','c_zip_724.0','c_zip_828.0','c_job_503','c_zip_843.0','c_occp_208','c_zip_720.0','c_zip_715.0','c_occp_607','c_occp_1102','c_occp_509','c_occp_407','c_zip_714.0','c_zip_713.0','c_zip_703.0','c_zip_655.0','c_zip_844.0','c_zip_653.0','c_zip_649.0','c_occp_701','c_occp_507','c_job_406','c_zip_716.0','c_zip_647.0','c_zip_646.0','c_zip_643.0','c_zip_845.0','c_zip_636.0','c_occp_604','c_occp_605','c_occp_606','c_zip_648.0','c_zip_719.0','c_occp_702','c_occp_704','c_zip_851.0','c_job_404','c_zip_635.0','c_zip_634.0','c_occp_804','c_zip_846.0','c_occp_806','c_zip_941.0','c_zip_633.0','c_zip_847.0','c_occp_705',
'c_occp_810','c_zip_621.0','c_zip_624.0','c_job_310','c_occp_815','c_zip_852.0','c_zip_623.0','c_zip_880.0','c_occp_819','c_job_307','c_job_305','c_zip_625.0','c_occp_902','c_zip_616.0','c_zip_821.0','c_occp_1101','c_zip_893.0','c_zip_615.0','c_zip_614.0','c_zip_882.0','c_zip_613.0','c_job_301','c_zip_883.0','c_zip_884.0','c_zip_611.0','c_zip_885.0','c_occp_1008','c_occp_1010','c_occp_903','c_zip_607.0','c_zip_605.0','c_zip_894.0','c_zip_901.0','c_occp_1015','c_zip_902.0','c_zip_903.0','c_zip_905.0','c_occp_1019','c_zip_909.0','c_zip_556.0','c_zip_971.0','c_zip_267.0','c_zip_927.0','c_zip_509.0','c_zip_912.0','c_zip_555.0','c_zip_913.0','c_zip_553.0','c_zip_552.0','c_zip_551.0','c_zip_921.0','c_zip_545.0','c_zip_507.0','c_zip_544.0','c_zip_541.0','c_zip_922.0','c_zip_530.0','c_zip_528.0','c_zip_527.0','c_zip_526.0','c_zip_525.0','c_zip_524.0','c_zip_523.0','c_zip_983.0','c_zip_542.0','c_zip_515.0','c_zip_923.0','c_zip_362.0','c_job_501','c_zip_815.0','c_zip_975.0','c_zip_504.0','c_zip_979.0','c_zip_502.0','c_zip_439.0','c_zip_978.0','c_zip_426.0','c_zip_424.0','c_zip_972.0','c_zip_977.0','py_l','c_zip_925.0','c_zip_400.0','c_zip_369.0','c_zip_368.0','c_zip_926.0','c_zip_366.0','c_zip_365.0','c_zip_364.0','c_zip_363.0','c_zip_976.0','c_zip_513.0','c_zip_982.0','c_job_303','c_zip_357.0','c_zip_312.0','c_occp_1103','c_zip_353.0','c_zip_965.0','c_zip_336.0','c_zip_964.0','c_zip_963.0','c_zip_962.0','c_zip_961.0','c_zip_264.0','c_zip_315.0','c_zip_311.0','c_zip_954.0','c_zip_959.0','c_zip_958.0','c_zip_957.0','c_zip_956.0','c_zip_272.0','c_zip_270.0','c_zip_955.0','c_zip_268.0','c_zip_313.0','c_zip_953.0','c_zip_942.0','c_zip_943.0','c_zip_354.0','c_zip_966.0','c_zip_227.0','c_zip_261.0','c_zip_952.0','c_zip_253.0','c_zip_951.0','c_zip_947.0','c_zip_233.0','c_zip_232.0','c_zip_931.0','c_zip_945.0','c_zip_228.0','c_zip_226.0','c_zip_263.0','c_zip_223.0','c_zip_212.0','c_zip_944.0','c_zip_211.0','c_zip_210.0','c_zip_209.0','c_zip_208.0','c_zip_205.0','c_zip_266.0','c_zip_940.0'})
```

##### 輸出檔案

```python
train_all.to_csv("sr_2_4_fund_sr_3_46_feature.csv")
```