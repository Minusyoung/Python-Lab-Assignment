import re
import time
import torch
import pickle
import numpy as np
from collections import deque
from mcrcon import MCRcon
from train import GameLSTM
from dataset import CustomMinMaxScaler

RCON_IP = "127.0.0.1"
RCON_PASS = "6567"
PLAYER_NAME = "Nickyaso"
WINDOW_SIZE = 20


def run_ai_agent():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    model = GameLSTM().to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    history_buffer = deque(maxlen=WINDOW_SIZE)
    last_loaded_chunk = None
    last_x, last_z = None, None

    print("=== 6D-Delta 在线流送调度代理已启动 ===")

    try:
        with MCRcon(RCON_IP, RCON_PASS) as mcr:
            mcr.command("gamerule sendCommandFeedback false")

            while True:
                loop_start = time.perf_counter()

                raw_pos = mcr.command(f"data get entity {PLAYER_NAME} Pos")
                match_pos = re.search(r'\[(.*?)d, (.*?)d, (.*?)d\]', raw_pos)
                raw_rot = mcr.command(f"data get entity {PLAYER_NAME} Rotation")
                match_rot = re.search(r'\[(.*?)f, (.*?)f\]', raw_rot)

                if match_pos and match_rot:
                    x, y, z = [float(match_pos.group(i)) for i in range(1, 4)]
                    yaw, pitch = float(match_rot.group(1)), float(match_rot.group(2))

                    v_xz = 0.0 if last_x is None else np.sqrt((x - last_x) ** 2 + (z - last_z) ** 2)
                    last_x, last_z = x, z

                    history_buffer.append([x, y, z, yaw, pitch, v_xz])

                    if len(history_buffer) == WINDOW_SIZE:
                        recent_data = np.array(history_buffer)
                        scaled_data = scaler_X.transform(recent_data)
                        tensor_data = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)

                        with torch.no_grad():
                            pred_delta_scaled = model(tensor_data).cpu().numpy()

                        # 逆归一化相对位移，并加回当前坐标
                        pred_delta = scaler_y.inverse_transform(pred_delta_scaled)[0]
                        pred_x = x + pred_delta[0]
                        pred_z = z + pred_delta[1]

                        chunk_x, chunk_z = int(pred_x // 16), int(pred_z // 16)

                        if (chunk_x, chunk_z) != last_loaded_chunk:
                            print(f"航速: {v_xz:.1f} 块/帧 | ⚡ 预载区块: [{chunk_x}, {chunk_z}]")
                            mcr.command("forceload remove all")
                            mcr.command(f"forceload add {chunk_x} {chunk_z}")
                            last_loaded_chunk = (chunk_x, chunk_z)

                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.1: time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        print("\nAI 代理安全关闭。")


if __name__ == "__main__":
    run_ai_agent()