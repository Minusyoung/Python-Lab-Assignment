import pandas as pd
import numpy as np

df = pd.read_csv('training_data.csv')
df['v_x'] = df['x'].diff().fillna(0)
df['v_z'] = df['z'].diff().fillna(0)
df['v_xz'] = np.sqrt(df['v_x']**2 + df['v_z']**2)

# 假设 10Hz 采样下，每帧移动超过 1.0 个区块视为飞行 (即 10区块/秒以上)
flight_data = df[df['v_xz'] > 1.0]
walk_data = df[df['v_xz'] <= 1.0]

print(f"总数据量: {len(df)}")
print(f"飞行数据占比: {len(flight_data) / len(df) * 100:.2f}%")
print(f"行走/低速数据占比: {len(walk_data) / len(df) * 100:.2f}%")