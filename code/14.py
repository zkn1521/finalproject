import pandas as pd
import numpy as np
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
df_training_set = pd.read_csv('aa4_x_training_set.csv')


# print(df_training_set)
# 标准化方法1
# df_training_set_ = preprocessing.scale(df_training_set)
# print(preprocessing.scale(df_training_set))

# 标准化方法2
# df_training_set_mean = df_training_set.mean(axis=0)
# df_training_set_std = df_training_set.std(axis=0)
# X = (df_training_set - df_training_set_mean) / df_training_set_std
# print(X)

# 标准化方法3
# scaler = preprocessing.StandardScaler()
# X_scaled = scaler.fit_transform(df_training_set)
# print(X_scaled)
# data_df = pd.DataFrame(X_scaled)
# print(data_df)
# print(data_df.std(axis=0))

# origin_data = scaler.inverse_transform(X_scaled)
# print(origin_data)

# 归一化
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(df_training_set)
data_df = pd.DataFrame(X_scaled)
# print(data_df)


data_df.to_csv('aa4_x_training_set_normalization.csv', encoding='utf-8', index=False,
               columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
