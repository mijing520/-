import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from scipy.interpolate import interp1d
from scipy import interpolate

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_x = pd.read_excel(r'E:\风功率\数据\2019年功率数据和天气预报数据\碧柳河\20190101-20190522-东电茂霖（经棚）碧柳河风电场-碧柳河集电线-天气预报信息报表.xls')
# print(np.any(pd.isnull(data_x)))  # 返回false
data_y = pd.read_excel(r'E:\风功率\数据\2019年功率数据和天气预报数据\碧柳河\20190101-20190521-东电茂霖（经棚）碧柳河风电场-碧柳河集电线-功率报表.xls')
# print(np.any(pd.isnull(data_y)))  # 返回true
# print(pd.isnull(data_y))  # 返回true

data0 = pd.merge(data_x,data_y,on=['日期','时间'],how='inner')
print(data0.shape)
data0 = data0[["日期", "时间","风速(m/s)","风向(°)","湿度(RH)","温度(℃)","气压(KPa)","空气密度(kg/m³)","实际功率(MW)"]]
data0['time']=data0['日期']+' '+data0['时间']
data0['time']=pd.to_datetime(data0['time'])
data0.set_index(data0['time'],inplace=True)
data0.drop(['日期','时间','time'],axis=1,inplace=True)
# print(data0.head())
# print(data0.head())
# print(data0.shape)
# print(pd.isnull(data0["实际功率(MW)"]))
data1 = data0.interpolate(method='linear')  # interpolate 中只有一个linear 方法 是为什么  默认好像就是linear
# print(pd.isnull(data1["实际功率(MW)"]))
# print(data1.tail())
print(data1.tail())

x_cols = [c for c in data1.columns if c !="实际功率(MW)"]
x = data1[x_cols]
y=data1["实际功率(MW)"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=666)

# 训练随机森林算法
rf = RandomForestRegressor()
rf = rf.fit(x_train,y_train)

# 训练 极端随机树
et = ExtraTreesRegressor()
et = et.fit(x_train,y_train)

# 保存
joblib.dump(rf, "./random_forest.joblib", compress=True)
joblib.dump(et, "./extra_trees.joblib", compress=True)
# interpld 不太会用
# data1 = interp1d(data0["实际功率(MW)"],kind='quadratic')
# print(pd.isnull(data1))


# data1 = data0["实际功率(MW)"]
# data1.plot()
# plt.show()

# for 循环填充缺失值
# for i in data0.columns:
#     if np.any(pd.isnull(data0[i]))==True:
#         data0[i].fillna(value=data0[i].mean,inplace=True)
# print(np.any(pd.isnull(data0)))  # 返回false


# data1=data0['湿度(RH)']
# print(data1.head())
# data1.plot()
# plt.show()
