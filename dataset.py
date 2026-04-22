import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random


class CustomMinMaxScaler:
    def __init__(self):
        self.data_min = None;
        self.data_max = None

    def fit_transform(self, data):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        range_diff = self.data_max - self.data_min
        range_diff[range_diff == 0] = 1e-8
        return (data - self.data_min) / range_diff

    def transform(self, data):
        range_diff = self.data_max - self.data_min
        range_diff[range_diff == 0] = 1e-8
        return (data - self.data_min) / range_diff

    def inverse_transform(self, data):
        return data * (self.data_max - self.data_min) + self.data_min


class TrajectoryDataset(Dataset):
    def __init__(self, csv_path, window_size=20, horizon=5, is_train=True):
        self.window_size = window_size
        self.horizon = horizon

        df = pd.read_csv(csv_path)

        # 1. 计算输入特征：瞬时速度 V_xz
        df['v_x'] = df['x'].diff().fillna(0)
        df['v_z'] = df['z'].diff().fillna(0)
        df['v_xz'] = np.sqrt(df['v_x'] ** 2 + df['v_z'] ** 2)

        # 2. 【核心重构】：计算相对位移 (Delta) 作为完美预测目标
        df['delta_target_x'] = df['x'].shift(-horizon) - df['x']
        df['delta_target_z'] = df['z'].shift(-horizon) - df['z']
        df = df.dropna().reset_index(drop=True)

        # 6D 输入与 2D Delta 目标
        features_6d = df[['x', 'y', 'z', 'yaw', 'pitch', 'v_xz']].values
        targets_deltas = df[['delta_target_x', 'delta_target_z']].values

        if is_train:
            self.scaler_X = CustomMinMaxScaler()
            self.scaler_y = CustomMinMaxScaler()
            features_scaled = self.scaler_X.fit_transform(features_6d)
            targets_scaled = self.scaler_y.fit_transform(targets_deltas)
            with open('scaler_X.pkl', 'wb') as f:
                pickle.dump(self.scaler_X, f)
            with open('scaler_y.pkl', 'wb') as f:
                pickle.dump(self.scaler_y, f)
        else:
            with open('scaler_X.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)
            features_scaled = self.scaler_X.transform(features_6d)
            targets_scaled = self.scaler_y.transform(targets_deltas)

        self.X, self.y = [], []
        random.seed(42)
        drop_count, keep_count = 0, 0

        # 3. 滑动窗口与动态欠采样
        for i in range(len(features_scaled) - window_size):
            window_real_speeds = features_6d[i: i + window_size, 5]
            avg_speed = np.mean(window_real_speeds)

            # 低速冗余数据 75% 概率丢弃
            if is_train and avg_speed <= 1.0 and random.random() < 0.75:
                drop_count += 1
                continue

            self.X.append(features_scaled[i: i + window_size])
            self.y.append(targets_scaled[i + window_size - 1])  # Delta 已经是对齐好的
            keep_count += 1

        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

        if is_train:
            print(f"📊 数据均衡处理完毕！丢弃冗余 {drop_count} 个，保留有效 {keep_count} 个")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]