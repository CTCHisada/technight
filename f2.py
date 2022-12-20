from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# 線形回帰モデルのLinearRegressionをインポート
from sklearn.linear_model import LinearRegression
# 時系列分割のためTimeSeriesSplitのインポート
from sklearn.model_selection import TimeSeriesSplit
# 予測精度検証のためMSEをインポート
from sklearn.metrics import mean_squared_error as mse


###データ取得・加工###
# pandas_datareaderを使って、2022年のctc株価データの取得
S='2018-01-01'
E ='2022-12-20'

import yfinance as yf
# pandas_datareaderを使って、2022年のctc株価データの取得
#data_master = data.DataReader('4739.T', 'yahoo', S, E)
#yfinanceを使って、2022年のctc株価データの取得
data_master = yf.download('4739.T', S, E)
print(data_master)



# 曜日情報を追加(月曜:0, 火曜:1, 水曜:2, 木曜:3, 金曜:4、土曜:5、日曜:6)
data_master['weekday'] = data_master.index.weekday

###データ取得・加工###
# 移動平均を追加
# data_techinicalにデータをコピー
data_technical = data_master.copy()

SMA1 = 5   #短期5日
SMA2 = 10  #中期10日
SMA3 = 15  #長期15日
data_technical['SMA1'] = data_technical['Close'].rolling(SMA1).mean() #短期移動平均の算出
data_technical['SMA2'] = data_technical['Close'].rolling(SMA2).mean() #中期移動平均の算出
data_technical['SMA3'] = data_technical['Close'].rolling(SMA3).mean() #長期移動平均の算出

# OpenとCloseの差分を実体Bodyとして計算
data_technical['Body'] = data_technical['Open'] - data_technical['Close']
# 前日終値との差分Close_diffを計算
data_technical['Close_diff'] = data_technical['Close'].diff(1)
# 目的変数となる翌日の終値Close_nextの追加
data_technical['Close_next'] = data_technical['Close'].shift(-1)

# 欠損値がある行を削除
data_technical = data_technical.dropna(how='any')

# 必要なカラムを抽出
data_technical = data_technical[['High', 'Low', 'Open', 'Close', 'Body','Close_diff', 'SMA1', 'SMA2', 'SMA3', 'Close_next']]

print(data_technical)

#分析
# 2018年〜2021年を学習用データとする
train = data_technical['2018-01-01' : '2021-12-31']

# 2022年をテストデータとする
test = data_technical['2022-01-01' :]

# 学習用データとテストデータそれぞれを説明変数と目的変数に分離する
#説明変数：目的変数を説明する変数 y=ax+b のx
#目的変数：予測したい変数 y=ax+b のy

X_train = train.drop(columns=['Close_next']) #学習用データ説明変数
y_train = train['Close_next'] #学習用データ目的変数
X_test = test.drop(columns=['Close_next']) #テストデータ説明変数
y_test = test['Close_next'] #テストデータ目的変数

# 時系列分割交差検証
valid_scores = []
tscv = TimeSeriesSplit(n_splits=4)
for fold, (train_indices, valid_indices) in enumerate(tscv.split(X_train)):
    X_train_cv, X_valid_cv = X_train.iloc[train_indices], X_train.iloc[valid_indices]
    y_train_cv, y_valid_cv = y_train.iloc[train_indices], y_train.iloc[valid_indices]
    # 線形回帰モデルのインスタンス化
    model = LinearRegression()
    # モデル学習
    model.fit(X_train_cv, y_train_cv)
    # 予測
    y_valid_pred = model.predict(X_valid_cv)
    # 予測精度(RMSE)の算出
    score = np.sqrt(mse(y_valid_cv, y_valid_pred))
    # 予測精度スコアをリストに格納
    valid_scores.append(score)

#誤差のスコア
print(f'valid_scores: {valid_scores}')
cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score}')

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = np.sqrt(mse(y_test, y_pred))
print(f'RMSE: {score}')

# 実際のデータと予測データをデータフレームにまとめる
df_result = test[['Close_next']]
df_result['Close_pred'] = y_pred
# 誤差を算出
df_result['diff'] = df_result['Close_pred'] - df_result['Close_next']

print(df_result)

# 実際のデータと予測データの比較グラフ作成
plt.figure(figsize=(15, 6))
plt.xticks()
plt.plot(df_result['Close_next'], label='Close_next', color='orange')
plt.plot(df_result['Close_pred'], label='Close_pred', color='blue')
plt.xlabel('Date')
plt.ylabel('JPY')
xmin = df_result.index.min()
xmax = df_result.index.max()
plt.legend()
plt.show()
