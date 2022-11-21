# 데이터 분석 툴
import pandas as pd
import numpy as np

# 운영체계와 관련된 툴
import os
import glob

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

# 경고 방지
import warnings

warnings.filterwarnings('ignore')

# ?
plt.figure(figsize=(15, 15))
plt.style.use('ggplot')

df = pd.read_csv("./data/kemp-abh-sensor.csv", index_col="Index")
df_OK = df[df["NG"] == 0]
df_NG = df[df["NG"] == 1]
print(df)
print(df_OK.iloc[:, 1:4])
print(df_NG)

# 일반적인 데이터 분포 확인
plt.scatter(df_OK["Temp"], df_OK["Current"], color="limegreen")
plt.scatter(df_NG["Temp"], df_NG["Current"], color="red")
plt.axhline(1.4, color="dodgerblue")
plt.axvline(95, color="dodgerblue")
plt.xlabel("Temperature")
plt.ylabel("Voltage")
plt.show()

# 상관 계수
corr = df.iloc[:, 1:4].corr()
sns.heatmap(corr, annot=True, cmap="Greens", annot_kws={"size": 20})
plt.title("corr")
plt.show()

corr = df_OK.iloc[:, :3].corr()
sns.heatmap(corr, annot=True, cmap="Blues", annot_kws={"size": 20})
plt.title("corr")
plt.show()

corr = df_NG.iloc[:, :3].corr()
sns.heatmap(corr, annot=True, cmap="Reds", annot_kws={"size": 20})
plt.title("corr")
plt.show()

pick_data = df

# 학습, 평가 데이터 준비
# 데이터 분리
train_data = pick_data[:35604]
test_data = pick_data[35604:]

# 정상 데이터
# ng_idx_train = train_data[train_data['NG'] == 1].index
ok_idx_train = train_data[train_data['NG'] == 0].index
tNc_ok_train = train_data.loc[ok_idx_train]
# tNc_ng_train = train_data.loc[ng_idx_train]

# 데이터 스케일링
from sklearn.preprocessing import StandardScaler

train = tNc_ok_train
scaler = StandardScaler()
scaler = scaler.fit(train[['Temp']])
train["Temp"] = scaler.transform(train[["Temp"]])
# print(train)

# 시계열성 데이터 분리
TIME_STEP = 36

def create_sequences(X, y, time_steps=TIME_STEP):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i:(i + time_steps)].values)
        #   ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)

X_train, Y_train = create_sequences(train[["Temp"]], train[["Temp"]])

# 데이터 분리 작업 진행 (7:3)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2)

ss_data=StandardScaler().fit_transform(test_data[['Temp']])
print(ss_data)
X_data = pd.DataFrame()
X_data['Temp'] = list(ss_data.reshape(-1))
X_test, Y_test = create_sequences(X_data[['Temp']], test_data[['NG']])


def flatten(X):
    # sample x features array.
    flattened_X = np.empty((X.shape[0 ], X.shape[2 ]))
    for i in range(X.shape[0 ]):
        flattened_X[i] = X[i, (X.shape[1 ]-1 ), :]
    return (flattened_X)


label = flatten(Y_test).reshape(-1)

timesteps = X_train.shape[1]
features = X_train.shape[2]

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras import backend as K
from keras.layers import *
from keras.applications import imagenet_utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

Lstm_AE_3 = Sequential()
# Encoder
Lstm_AE_3.add(LSTM(32, activation='relu', input_shape=(timesteps, features), return_sequences=True))
Lstm_AE_3.add(LSTM(16, activation='relu', return_sequences=False))
Lstm_AE_3.add(RepeatVector(timesteps))
# Decoder
Lstm_AE_3.add(LSTM(32, activation='relu', return_sequences=True))
Lstm_AE_3.add(LSTM(16, activation='relu', return_sequences=True))
Lstm_AE_3.add(TimeDistributed(Dense(features)))
Lstm_AE_3.summary()


# CPU 사용 경우
epochs = 5
batch = 128
lr = 0.001
optimizer = keras.optimizers.Adam(lr)
Lstm_AE_3.compile(loss='mse', optimizer=optimizer)
history = Lstm_AE_3.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_valid, y_valid))

# 결과표
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.xlabel('Epoch'); plt.ylabel('loss')
plt.show()

#
prediction = Lstm_AE_3.predict(X_test)
mse = np.mean(np.power(X_test - prediction, 2 ), axis=1 )
error_df = pd.DataFrame({'reconstruction_error': mse.reshape(-1 ),
             'true_class':label})

#
thr = np.percentile(mse.reshape(-1 ),75)

from sklearn.metrics import *
pred_y = [1 if e >thr else 0 for e in error_df['reconstruction_error'].values]
conf_matrix = confusion_matrix(error_df['true_class'], pred_y)
plt.figure(figsize=(7 , 7 ))
sns.heatmap(conf_matrix, annot=True , fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class'); plt.ylabel('True Class')
plt.show()

#
TP = conf_matrix[0 ][0 ]
FN = conf_matrix[0 ][1 ]
FP = conf_matrix[1 ][0 ]
TN = conf_matrix[0 ][1 ]
Recall = TP / (TP + FN)
Precision = TP / (TP + FP)
Accuracy = (TP + TN) / (TP + FP + FN + TN)
F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
print(Recall, Precision, Accuracy, F1_Score)