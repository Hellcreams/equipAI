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

# 경로 이름 설정
path_name = os.getcwd() + "/data" + "/5공정_180sec"

# 데이터 병합 코드
data_list = list()
cnt = 0

for pth in glob.glob(path_name + '/*')[1:]:
    origin = pd.read_csv(pth, index_col=False)
    origin["Date"] = "-".join(pth.split('-')[-1].split('.')[:-1])
    cnt = cnt + len(origin)
    data_list.append(origin)

Cat_list = pd.concat(data_list, axis=0, ignore_index=True)
# print(Cat_list)

# 사본 생성
df = Cat_list.copy()

# 데이터 분포 확인
"""
for i in df.columns:
    try:
        df[i].plot()
        plt.tight_layout()
        plt.legend()
        plt.show()
    except:
        pass
"""

# 데이터 추출
pick_data = df[df.columns[3:]]
print(pick_data.head(1))

# 상관 계수
corr = pick_data.corr()
sns.heatmap(corr, annot=True, cmap="Greens", annot_kws={"size": 20})
plt.title("corr")
plt.show()

# 데이터 정제(전처리)
df_er = pd.read_csv(glob.glob(path_name + '/*')[0], index_col=False)
df_err = df_er.transpose()
df_err.columns = df_err.iloc[0]
df_err = df_err.iloc[1:, :].fillna(0.0)
print(df_err.head())

# 무의미한 특성 제거
pick_data = df[df.columns[3:]]
# print(pick_data.head())
# print(df[df["Process"] == df_err['2021-09-06'][0]])

pick_data['NG'] = 0
for dt in df_err.columns:
    pick_DATE = df[df.Date == dt]
    for i in range(len(df_err)):  # len(df_err) : 11개 -> 한 날짜에 최대 에러 11개
        single_feature = df_err[dt][i]  # 에러 루트 위치 검출
        if single_feature != 0:
            c_inx = pick_DATE['Temp'][pick_DATE['Process'] == single_feature].index
            pick_data['NG'].loc[c_inx] = 1
        else:
            pass

# print(pick_data["NG"].value_counts())
# print(pick_data["NG"])


# 학습, 평가 데이터 준비
# 데이터 분리
train_data = pick_data[:35604]
test_data = pick_data[35604:]

# 정상 데이터
ng_idx_train = train_data[train_data['NG'] == 1].index
ok_idx_train = train_data[train_data['NG'] == 0].index
tNc_ok_train = train_data.loc[ok_idx_train]
tNc_ng_train = train_data.loc[ng_idx_train]

# 정상 데이터 2
from sklearn.preprocessing import StandardScaler

train = tNc_ok_train
scaler = StandardScaler()
scaler = scaler.fit(train[['Temp']])
train["Temp"] = scaler.transform(train[["Temp"]])

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
x_train, x_vaild, y_train, y_vaild = train_test_split(X_train, Y_train, test_size=0.2)


# 불량이 포함된 테스트 데이터 분리 코드
ss_data = StandardScaler().fit_transform(test_data[["Temp"]])
print(ss_data)