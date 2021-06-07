import mplfinance as mpf
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#直近1年間のデータより予測モデルの作成
start="2020-05-06"
end="2021-04-25"

df=data.DataReader("MXNJPY=X","yahoo",start,end)
df = df.drop(labels = ["Volume"],axis = 1)
df = df.drop(labels = ["Adj Close"],axis = 1)

#x:入力値 t:xに対する目的値
t = df["Close"].values
x = df.drop(labels = ["Close"],axis = 1)

#trainを学習用のデータ,testを作成したモデルに適応させるテストデータとする。
x_train,x_test,t_train,t_test = train_test_split(x,t,test_size = 0.3,random_state = 0)

#線形回帰モデルを使用
model = LinearRegression()
model.fit(x_train,t_train)

#
y = model.predict(x_test)

print(f"予測値: {y[0]}")
print(f"目標値: {t_test[0]}")

#作成した線形回帰モデルを使って、終値を予測する
predict_start="2021-04-26"
predict_end="2021-05-05"

predict_df=data.DataReader("MXNJPY=X","yahoo",predict_start,predict_end)
predict_df = predict_df.drop(labels = ["Volume"],axis = 1)
predict_df = predict_df.drop(labels = ["Adj Close"],axis = 1)

x_pre = predict_df.drop(labels = ["Close"],axis = 1)
t_pre = predict_df["Close"].values

y_pre = model.predict(x_pre)

list = []
i = 0

while i <= 7:
    list = list + [y_pre[i]]
    i += 1

predict_df["Predict"] = list

predict_df = predict_df.drop(labels = ["High"],axis = 1)
predict_df = predict_df.drop(labels = ["Low"],axis = 1)
predict_df = predict_df.drop(labels = ["Open"],axis = 1)
predict_df.plot()
grapf = predict_df.plot()

grapf.figure.savefig('output.png')
