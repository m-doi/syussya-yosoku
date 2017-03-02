%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
df1 = pd.read_csv("data_201702.csv", header=0)
df2 = pd.read_csv("data_201702.csv", header=0)
df2['day'] = df['day'] + 1
df3 = pd.merge(df1,  df2,  how='inner', on='day')

import numpy as np

df3['avg_minus'] = df3['avgdc_x'] - df3['avgdc_y']
df4 = df3.ix[:, ['start_y', 'end_y',  'weekday_y', 'rest_x','weekday_x', 'avgdc_x', 'maxcd_x', 'mincd_x', 'avg_minus', 'start_x']]
df5 = df4[(df4['weekday_x'] != 1) & (df4['weekday_x'] != 7) & (df4['rest_x'] == 0)].ix[:, ['start_y', 'end_y',  'weekday_y','weekday_x', 'avgdc_x', 'maxcd_x', 'mincd_x', 'avg_minus', 'start_x']]

import datetime, time

def time2Float(hhmm):
    p = hhmm.split(':')
    return int(p[0]) + int(p[1])/60

df5['start_y'] = df5['start_y'].apply(lambda x:time2Float(str(x)), 0)
df5['end_y'] = df5['end_y'].apply(lambda x:time2Float(str(x)), 0)
df5['start_x'] = df5['start_x'].apply(lambda x:time2Float(str(x)), 0)

mins = df5.min()
maxes = df5.max()
means = df5.mean()

df_norm = (df5 - df5.mean()) / (df5.max() - df5.min())
df_norm.head(28)

def normalize(src, mean, max, min):
    return (src - mean) / (max - min)

def unNormalize(src, mean, max, min):
    return src * (max - min) + mean

# sklearn.linear_model.LinearRegression クラスを読み込み
from sklearn import linear_model
clf = linear_model.LinearRegression()

# 説明変数
X = df_norm.loc[:, ['start_y', 'end_y', 'weekday_y', 'weekday_x', 'avgdc_x', 'maxcd_x', 'mincd_x', 'avg_minus']].as_matrix()

# 目的変数
Y = df_norm['start_x'].as_matrix()

# 予測モデルを作成
clf.fit(X, Y)

# 回帰係数
# print(clf.coef_)

# 切片 (誤差)
# print(clf.intercept_)

# 決定係数
# print(clf.score(X, Y))

# matplotlib パッケージを読み込み
import matplotlib.pyplot as plt

# 散布図
# plt.scatter(X, Y)

# 回帰直線
plt.plot(X, clf.predict(X))

import math

# 予測
def hour2Time(hour):
    return str(math.floor(hour / 1)) + ':' + str(math.floor(hour % 1 * 60))


# print(hour2Time(9.844357002556329))

print(
    hour2Time(
        unNormalize(
            clf.predict(
        [
            normalize(9.00, means.get('start_y'), maxes.get('start_y'), means.get('start_y')),
            normalize(23.00, means.get('end_y'), maxes.get('end_y'), means.get('end_y')),
            normalize(5, means.get('weekday_y'), maxes.get('weekday_y'), means.get('weekday_y')),
            normalize(6, means.get('weekday_x'), maxes.get('weekday_x'), means.get('weekday_x')),
            normalize(10.0, means.get('avgdc_x'), maxes.get('avgdc_x'), means.get('avgdc_x')),
            normalize(13.0, means.get('maxcd_x'), maxes.get('maxcd_x'), means.get('maxcd_x')),
            normalize(8.0, means.get('mincd_x'), maxes.get('mincd_x'), means.get('mincd_x')),
            normalize(2.0, means.get('avg_minus'), maxes.get('avg_minus'), means.get('avg_minus'))
        ])[0],
            means.get('start_x'),
            maxes.get('start_x'),
            means.get('start_x')
        )
    )
)
